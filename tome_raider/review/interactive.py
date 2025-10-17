"""Interactive TUI for reviewing and editing dataset samples."""

from typing import List, Optional, Callable
from dataclasses import dataclass
from copy import deepcopy

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    DataTable, Header, Footer, Static, Button, Input, Label,
    TextArea, Select, Checkbox
)
from textual.binding import Binding
from textual.screen import Screen, ModalScreen
from textual import on
from textual.reactive import reactive

from loguru import logger

from ..sources.base import Sample
from ..storage.dataset_store import DatasetStore
from ..quality.validator import DatasetValidator


@dataclass
class ReviewStats:
    """Statistics for dataset review."""
    total: int = 0
    approved: int = 0
    rejected: int = 0
    pending: int = 0
    avg_quality: float = 0.0
    valid: int = 0
    invalid: int = 0


class HelpScreen(ModalScreen[None]):
    """Modal screen showing keyboard shortcuts and help."""

    CSS = """
    HelpScreen {
        align: center middle;
    }

    #help_container {
        width: 80;
        height: auto;
        max-height: 90%;
        background: $surface;
        border: thick $primary;
        padding: 2;
        overflow-y: auto;
    }

    .help_title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    .help_section {
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the help screen."""
        with Container(id="help_container"):
            yield Static("[bold cyan]Tome Raider - Keyboard Shortcuts[/bold cyan]", classes="help_title")

            help_text = """
[yellow]Navigation:[/yellow]
  ↑/↓ or k/j     - Move between rows
  Enter          - Open detail view for selected sample
  Escape         - Close modal/dialog

[yellow]Review Actions:[/yellow]
  a              - Approve selected sample
  r              - Reject selected sample
  d              - Delete selected sample

[yellow]View Options:[/yellow]
  p              - Toggle preview panel (shows full text)
  v              - Run validation on dataset

[yellow]File Operations:[/yellow]
  Ctrl+S         - Save changes to disk
  q or Ctrl+Q    - Quit application

[yellow]Detail View (when pressing Enter):[/yellow]
  - View full instruction and response text
  - Edit text directly (if not read-only)
  - See character/word/line counts
  - Scroll with ↑↓, PageUp/PageDown, Home/End
  - Press F2 to save edits and close
  - Press A to approve+close, R to reject+close
  - Press Escape to close without saving
  - Press Ctrl+S to write all changes to disk

[yellow]Preview Panel (press 'p' to enable):[/yellow]
  - Shows full text of currently selected sample
  - Updates automatically as you navigate
  - Scrollable for long content

[yellow]Tips:[/yellow]
  - Footer shows available shortcuts
  - Status bar shows dataset statistics
  - Changes marked with * (use Ctrl+S to save)
  - Read-only mode prevents accidental edits

Press [bold]?[/bold] anytime to show this help screen.
Press [bold]Escape[/bold] or [bold]q[/bold] to close.
"""
            yield Static(help_text, classes="help_section")

    def action_close(self) -> None:
        """Close the help screen."""
        self.dismiss()


class DetailViewScreen(ModalScreen[bool]):
    """Modal screen for viewing and editing sample details."""

    CSS = """
    DetailViewScreen {
        align: center middle;
    }

    #detail_container {
        width: 90%;
        height: 90%;
        background: $surface;
        border: thick $primary;
        padding: 1;
    }

    #instruction_area {
        height: 45%;
        border: solid $accent;
        margin-bottom: 1;
    }

    #response_area {
        height: 45%;
        border: solid $accent;
        margin-bottom: 1;
    }

    .text_stats {
        color: $text-muted;
        margin: 0 1;
    }

    .metadata_line {
        margin: 0 1;
    }

    .button_bar {
        height: 3;
        align: center middle;
    }

    Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("f2", "save", "Save"),
        Binding("a", "approve", "Approve"),
        Binding("r", "reject", "Reject"),
    ]

    def __init__(
        self,
        sample: Sample,
        sample_index: int,
        readonly: bool = False
    ):
        super().__init__()
        self.sample = deepcopy(sample)
        self.original_sample = sample
        self.sample_index = sample_index
        self.readonly = readonly
        self.modified = False

    def compose(self) -> ComposeResult:
        """Compose the detail view."""
        with Container(id="detail_container"):
            yield Static(f"Sample #{self.sample_index}", id="title")

            # Instruction with stats
            inst_lines = self.sample.instruction.count('\n') + 1
            inst_words = len(self.sample.instruction.split())
            yield Label(f"Instruction ({len(self.sample.instruction)} chars, {inst_words} words, {inst_lines} lines):")
            instruction_area = TextArea(
                self.sample.instruction,
                id="instruction_area",
                show_line_numbers=True,
                read_only=self.readonly,
                soft_wrap=True  # Enable word wrapping
            )
            yield instruction_area
            yield Static("Scroll: ↑↓ or PageUp/PageDown | Home/End for start/end", classes="text_stats")

            # Response with stats
            resp_lines = self.sample.response.count('\n') + 1
            resp_words = len(self.sample.response.split())
            yield Label(f"Response ({len(self.sample.response)} chars, {resp_words} words, {resp_lines} lines):")
            response_area = TextArea(
                self.sample.response,
                id="response_area",
                show_line_numbers=True,
                read_only=self.readonly,
                soft_wrap=True  # Enable word wrapping
            )
            yield response_area
            yield Static("Scroll: ↑↓ or PageUp/PageDown | Home/End for start/end", classes="text_stats")

            # Metadata
            quality = self.sample.metadata.quality_score or "Not scored"
            status = self.sample.metadata.review_status
            tags = ", ".join(self.sample.metadata.tags) if self.sample.metadata.tags else "None"

            yield Static(f"Quality: {quality} | Status: {status} | Tags: {tags}", classes="metadata_line")

            # Buttons
            with Horizontal(classes="button_bar"):
                if not self.readonly:
                    yield Button("Save [F2]", id="btn_save", variant="primary")
                    yield Button("Approve [A]", id="btn_approve", variant="success")
                    yield Button("Reject [R]", id="btn_reject", variant="error")
                yield Button("Cancel [Esc]", id="btn_close")

            # Footer to show keyboard shortcuts
            yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn_approve":
            self.action_approve()
        elif event.button.id == "btn_reject":
            self.action_reject()
        elif event.button.id == "btn_save":
            self.action_save()
        elif event.button.id == "btn_close":
            self.action_cancel()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Track modifications."""
        if not self.readonly:
            self.modified = True

    def action_approve(self) -> None:
        """Mark as approved and close."""
        self._save_changes()
        self.sample.metadata.review_status = "approved"
        self.dismiss()

    def action_reject(self) -> None:
        """Mark as rejected and close."""
        self._save_changes()
        self.sample.metadata.review_status = "rejected"
        self.dismiss()

    def action_save(self) -> None:
        """Save changes to memory and close."""
        logger.info("DetailViewScreen.action_save() called - saving changes and closing")
        self._save_changes()
        logger.info("DetailViewScreen.action_save() - about to call dismiss(True)")
        self.dismiss(True)
        logger.info("DetailViewScreen.action_save() - dismiss(True) called, method ending")

    def action_cancel(self) -> None:
        """Close without saving changes."""
        self.dismiss()

    def _save_changes(self) -> None:
        """Apply changes from text areas."""
        logger.info(f"_save_changes() called - readonly={self.readonly}")
        if not self.readonly:
            instruction_area = self.query_one("#instruction_area", TextArea)
            response_area = self.query_one("#response_area", TextArea)

            # Get text content from TextArea
            # In Textual, TextArea uses .text property to get/set content
            new_instruction = str(instruction_area.text)
            new_response = str(response_area.text)

            # Info logging so we can see it
            logger.info(f"Saving changes - Original instruction length: {len(self.original_sample.instruction)}")
            logger.info(f"Saving changes - New instruction length: {len(new_instruction)}")
            logger.info(f"Saving changes - Original response length: {len(self.original_sample.response)}")
            logger.info(f"Saving changes - New response length: {len(new_response)}")

            # Save to original sample (which is a reference to the dataset sample)
            self.original_sample.instruction = new_instruction
            self.original_sample.response = new_response

            logger.info(f"After save - Sample instruction length: {len(self.original_sample.instruction)}")
            logger.info(f"After save - Sample response length: {len(self.original_sample.response)}")
        else:
            logger.warning("_save_changes() called but readonly=True")


class PreviewPanel(Static):
    """Preview panel for viewing full text of selected sample."""

    CSS = """
    PreviewPanel {
        height: 30%;
        border-top: solid $primary;
        background: $surface;
        padding: 1;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }

    .preview_header {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    .preview_section {
        margin-bottom: 1;
    }

    .preview_label {
        color: $text-muted;
        text-style: italic;
    }
    """

    sample = reactive(None)

    def watch_sample(self, new_sample: Optional[Sample]) -> None:
        """Update display when sample changes."""
        if new_sample:
            self.display = True
            self.update_preview(new_sample)
        else:
            self.display = False

    def update_preview(self, sample: Sample) -> None:
        """Update preview content."""
        # Build preview text
        inst_lines = sample.instruction.count('\n') + 1
        inst_words = len(sample.instruction.split())
        resp_lines = sample.response.count('\n') + 1
        resp_words = len(sample.response.split())

        content = f"""[bold cyan]Preview - Press 'p' to toggle[/bold cyan]

[dim italic]Instruction ({len(sample.instruction)} chars, {inst_words} words, {inst_lines} lines):[/dim italic]
{sample.instruction}

[dim italic]Response ({len(sample.response)} chars, {resp_words} words, {resp_lines} lines):[/dim italic]
{sample.response}

[dim italic]Metadata:[/dim italic]
Quality: {sample.metadata.quality_score or 'Not scored'} | Status: {sample.metadata.review_status} | Tags: {', '.join(sample.metadata.tags) if sample.metadata.tags else 'None'}
"""

        # Update with rich text
        from rich.text import Text
        self.update(content)


class StatsBar(Static):
    """Statistics bar showing dataset info."""

    stats = reactive(ReviewStats())

    def watch_stats(self, new_stats: ReviewStats) -> None:
        """Update display when stats change."""
        self.update(self.render_stats(new_stats))

    def render_stats(self, stats: ReviewStats) -> str:
        """Render statistics text."""
        return (
            f"Total: {stats.total} | "
            f"✓ Approved: {stats.approved} | "
            f"✗ Rejected: {stats.rejected} | "
            f"⏸ Pending: {stats.pending} | "
            f"Avg Quality: {stats.avg_quality:.3f} | "
            f"Valid: {stats.valid}/{stats.total}"
        )


class DatasetReviewApp(App[None]):
    """Interactive TUI application for dataset review."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #main_table {
        height: 1fr;
    }

    #stats_bar {
        height: 1;
        background: $boost;
        color: $text;
        padding: 0 2;
    }

    #filter_bar {
        height: 3;
        background: $surface;
        padding: 0 1;
    }

    .status_approved {
        color: $success;
    }

    .status_rejected {
        color: $error;
    }

    .status_pending {
        color: $warning;
    }
    """

    BINDINGS = [
        Binding("a", "approve", "Approve"),
        Binding("r", "reject", "Reject"),
        Binding("d", "delete", "Delete"),
        Binding("p", "toggle_preview", "Preview"),
        Binding("v", "validate", "Validate"),
        Binding("ctrl+s", "save", "Save"),
        Binding("q", "quit", "Quit"),
        Binding("question_mark", "help", "Help"),
    ]

    TITLE = "Tome Raider - Dataset Review"

    def __init__(
        self,
        dataset_name: str,
        dataset: List[Sample],
        store: DatasetStore,
        readonly: bool = False,
        auto_validate: bool = False
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.store = store
        self.readonly = readonly
        self.auto_validate = auto_validate
        self.modified = False
        self.validator = DatasetValidator() if auto_validate else None
        self.current_filter = None
        self.preview_visible = False

    def compose(self) -> ComposeResult:
        """Compose the app layout."""
        yield Header()

        # Data table
        yield DataTable(id="main_table", zebra_stripes=True, cursor_type="row")

        # Preview panel (initially hidden)
        preview = PreviewPanel(id="preview_panel")
        preview.display = False
        yield preview

        # Stats bar
        yield StatsBar(id="stats_bar")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the table on mount."""
        table = self.query_one("#main_table", DataTable)

        # Add columns
        table.add_column("ID", width=6)
        table.add_column("Instruction", width=40)
        table.add_column("Response", width=40)
        table.add_column("Quality", width=10)
        table.add_column("Status", width=12)
        table.add_column("Tags", width=20)

        # Populate table
        self._populate_table()

        # Update stats
        self._update_stats()

        # Focus table
        table.focus()

        # Show startup notification
        self.notify(
            "Press [bold]?[/bold] for help | Check footer for keyboard shortcuts",
            severity="information",
            timeout=5
        )

    def _populate_table(self) -> None:
        """Populate the table with dataset samples."""
        table = self.query_one("#main_table", DataTable)
        table.clear()

        for idx, sample in enumerate(self.dataset):
            # Apply filter if active
            if self.current_filter:
                if self.current_filter not in sample.metadata.review_status:
                    continue

            # Truncate text for display
            instruction_preview = sample.instruction[:80] + "..." if len(sample.instruction) > 80 else sample.instruction
            response_preview = sample.response[:80] + "..." if len(sample.response) > 80 else sample.response

            # Format quality
            quality = f"{sample.metadata.quality_score:.3f}" if sample.metadata.quality_score else "-"

            # Format status with emoji
            status_emoji = {
                "approved": "✓",
                "rejected": "✗",
                "pending": "⏸"
            }.get(sample.metadata.review_status, "?")

            status = f"{status_emoji} {sample.metadata.review_status.capitalize()}"

            # Format tags
            tags = ", ".join(sample.metadata.tags[:3]) if sample.metadata.tags else ""

            # Add row
            table.add_row(
                str(idx),
                instruction_preview,
                response_preview,
                quality,
                status,
                tags,
                key=str(idx)
            )

    def _update_stats(self) -> None:
        """Update statistics bar."""
        stats = ReviewStats()
        stats.total = len(self.dataset)

        quality_scores = []

        for sample in self.dataset:
            status = sample.metadata.review_status
            if status == "approved":
                stats.approved += 1
            elif status == "rejected":
                stats.rejected += 1
            else:
                stats.pending += 1

            if sample.metadata.quality_score:
                quality_scores.append(sample.metadata.quality_score)

        if quality_scores:
            stats.avg_quality = sum(quality_scores) / len(quality_scores)

        # Validate if auto-validate is enabled
        if self.validator:
            validation_result = self.validator.validate_all(self.dataset)
            stats.valid = validation_result["valid"]
            stats.invalid = validation_result["invalid"]
        else:
            stats.valid = stats.total

        # Update the stats bar
        stats_bar = self.query_one("#stats_bar", StatsBar)
        stats_bar.stats = stats

    def _get_selected_sample(self) -> Optional[tuple[int, Sample]]:
        """Get the currently selected sample."""
        table = self.query_one("#main_table", DataTable)

        if table.cursor_row is None or table.cursor_row < 0:
            return None

        row_key = table.get_row_at(table.cursor_row)[0]
        idx = int(row_key)

        if 0 <= idx < len(self.dataset):
            return idx, self.dataset[idx]

        return None

    def _refresh_table_row(self, idx: int) -> None:
        """Refresh a specific table row from the current dataset state in memory.

        This method always updates the table display to match the in-memory dataset,
        ensuring synchronous consistency between data and display.

        Args:
            idx: Index of the sample to refresh in the table
        """
        if idx < 0 or idx >= len(self.dataset):
            logger.warning(f"Cannot refresh row {idx} - out of bounds")
            return

        # Get the current sample from dataset (the source of truth)
        updated_sample = self.dataset[idx]

        logger.debug(f"Refreshing table row {idx} - instruction length: {len(updated_sample.instruction)}")
        logger.debug(f"Refreshing table row {idx} - response length: {len(updated_sample.response)}")

        # Get table reference
        table = self.query_one("#main_table", DataTable)

        # Format the updated data
        instruction_preview = updated_sample.instruction[:80] + "..." if len(updated_sample.instruction) > 80 else updated_sample.instruction
        response_preview = updated_sample.response[:80] + "..." if len(updated_sample.response) > 80 else updated_sample.response
        quality = f"{updated_sample.metadata.quality_score:.3f}" if updated_sample.metadata.quality_score else "-"

        status_emoji = {
            "approved": "✓",
            "rejected": "✗",
            "pending": "⏸"
        }.get(updated_sample.metadata.review_status, "?")
        status = f"{status_emoji} {updated_sample.metadata.review_status.capitalize()}"

        tags = ", ".join(updated_sample.metadata.tags[:3]) if updated_sample.metadata.tags else ""

        # Update each cell in the row
        row_key = str(idx)
        table.update_cell(row_key, "ID", str(idx))
        table.update_cell(row_key, "Instruction", instruction_preview)
        table.update_cell(row_key, "Response", response_preview)
        table.update_cell(row_key, "Quality", quality)
        table.update_cell(row_key, "Status", status)
        table.update_cell(row_key, "Tags", tags)

        # Update stats (in case status changed)
        self._update_stats()

        # Update preview panel if visible
        if self.preview_visible:
            preview = self.query_one("#preview_panel", PreviewPanel)
            preview.sample = updated_sample

        logger.info(f"Table row {idx} refreshed from memory")

    def action_view_detail(self) -> None:
        """View sample details in modal.

        This method implements a synchronous flow where the table is ALWAYS refreshed
        from memory after the detail modal closes, regardless of how it was closed.
        This ensures the display always matches the in-memory dataset state.
        """
        selected = self._get_selected_sample()
        if not selected:
            return

        idx, sample = selected

        # Store original values to detect changes (for setting modified flag only)
        original_instruction = sample.instruction
        original_response = sample.response
        original_status = sample.metadata.review_status

        logger.info(f"Opening detail view for sample {idx}")

        # Define callback for when modal closes
        def handle_modal_close(result):
            logger.info(f"Modal close callback - result={result}")

            # Check if data actually changed to set the modified flag
            current_sample = self.dataset[idx]
            data_changed = (
                current_sample.instruction != original_instruction or
                current_sample.response != original_response or
                current_sample.metadata.review_status != original_status
            )

            logger.info(f"Data changed: {data_changed}")
            logger.info(f"Result from modal: {result}")

            # Set modified flag if data changed OR if F2 was pressed (result=True)
            if data_changed or result:
                self.modified = True
                logger.info(f"self.modified set to True (data_changed={data_changed}, result={result})")

            # ALWAYS repopulate the entire table from memory (synchronous approach)
            # This ensures the display always matches the in-memory state
            logger.info("Repopulating table from memory (synchronous refresh)")

            # Get table and store cursor position
            table = self.query_one("#main_table", DataTable)
            old_cursor_row = table.cursor_row

            # Repopulate table
            self._populate_table()
            self._update_stats()

            # Restore cursor position and force refresh
            if old_cursor_row is not None and old_cursor_row < table.row_count:
                table.move_cursor(row=old_cursor_row)
            table.refresh()

        # Show detail view with callback
        logger.info("About to call push_screen with callback...")
        self.push_screen(
            DetailViewScreen(sample, idx, self.readonly),
            callback=handle_modal_close
        )

    def action_approve(self) -> None:
        """Approve selected sample."""
        if self.readonly:
            return

        selected = self._get_selected_sample()
        if not selected:
            return

        _, sample = selected
        sample.metadata.review_status = "approved"
        self.modified = True
        self._populate_table()
        self._update_stats()

    def action_reject(self) -> None:
        """Reject selected sample."""
        if self.readonly:
            return

        selected = self._get_selected_sample()
        if not selected:
            return

        _, sample = selected
        sample.metadata.review_status = "rejected"
        self.modified = True
        self._populate_table()
        self._update_stats()

    def action_delete(self) -> None:
        """Delete selected sample."""
        if self.readonly:
            return

        selected = self._get_selected_sample()
        if not selected:
            return

        idx, _ = selected
        del self.dataset[idx]
        self.modified = True
        self._populate_table()
        self._update_stats()

    def action_validate(self) -> None:
        """Run validation on dataset."""
        if not self.validator:
            self.validator = DatasetValidator()

        self._update_stats()
        self.notify("Validation complete", severity="information")

    def action_save(self) -> None:
        """Save all in-memory changes to disk."""
        logger.debug(f"action_save called - self.modified = {self.modified}")
        logger.debug(f"action_save called - readonly = {self.readonly}")

        if self.readonly:
            self.notify("Read-only mode - cannot save", severity="warning")
            return

        # Save to disk
        if not self.modified:
            logger.warning("No changes to save - self.modified is False")
            self.notify("No changes to save", severity="information")
            return

        try:
            logger.info(f"Saving dataset with {len(self.dataset)} samples to disk")
            self.store.save(self.dataset, self.dataset_name)
            self.modified = False
            self.notify("Dataset saved to disk successfully", severity="information")
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            self.notify(f"Save failed: {e}", severity="error")

    def action_toggle_preview(self) -> None:
        """Toggle preview panel visibility."""
        preview = self.query_one("#preview_panel", PreviewPanel)
        table = self.query_one("#main_table", DataTable)

        self.preview_visible = not self.preview_visible
        preview.display = self.preview_visible

        if self.preview_visible:
            # Update preview with current selection
            selected = self._get_selected_sample()
            if selected:
                _, sample = selected
                preview.sample = sample
            self.notify("Preview panel enabled - use 'p' to toggle", severity="information")
        else:
            self.notify("Preview panel disabled", severity="information")

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Update preview when row selection changes."""
        if self.preview_visible:
            selected = self._get_selected_sample()
            if selected:
                _, sample = selected
                preview = self.query_one("#preview_panel", PreviewPanel)
                preview.sample = sample

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle Enter key press on a row - open detail view."""
        self.action_view_detail()

    def action_help(self) -> None:
        """Show help screen."""
        self.push_screen(HelpScreen())

    def action_quit(self) -> None:
        """Quit the application."""
        if self.modified:
            # TODO: Add confirmation dialog
            self.notify("Unsaved changes - use Ctrl+S to save first", severity="warning")
            return

        self.exit()


def launch_review_ui(
    dataset_name: str,
    dataset: List[Sample],
    store: DatasetStore,
    readonly: bool = False,
    auto_validate: bool = False
) -> None:
    """
    Launch the interactive review UI.

    Args:
        dataset_name: Name of the dataset
        dataset: List of samples
        store: Dataset store for saving
        readonly: Read-only mode
        auto_validate: Run validation automatically
    """
    app = DatasetReviewApp(
        dataset_name=dataset_name,
        dataset=dataset,
        store=store,
        readonly=readonly,
        auto_validate=auto_validate
    )

    app.run()
