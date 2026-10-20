/**
 * ApiKeyModal: opens on first run (no stored key), opens on the
 * `neurosight:unauthorized` window event, saves via setApiKey, closes on
 * Escape, and focuses the password input when opened.
 */
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { setApiKey, UNAUTHORIZED_EVENT } from "../api/client";
import { ApiKeyModal } from "./ApiKeyModal";

async function openModal() {
  render(<ApiKeyModal />);
  return waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument());
}

beforeEach(() => {
  // setApiKey always records the key in the client's in-memory fallback, so
  // reset it before clearing persisted storage — otherwise a key saved by an
  // earlier test would keep the modal closed for a "first run" scenario.
  setApiKey("");
  localStorage.clear();
});

describe("ApiKeyModal", () => {
  it("is hidden when a key already exists in localStorage", () => {
    localStorage.setItem("neurosight.apiKey", "existing-key");
    render(<ApiKeyModal />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("opens on mount when no key is stored and focuses the password input", async () => {
    await openModal();
    const input = screen.getByLabelText("API key");
    expect(input).toBeInTheDocument();
    expect(input).toHaveFocus();
  });

  it("opens when the unauthorized event is dispatched", async () => {
    localStorage.setItem("neurosight.apiKey", "existing-key");
    render(<ApiKeyModal />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    await act(async () => {
      window.dispatchEvent(new CustomEvent(UNAUTHORIZED_EVENT));
    });
    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument());
  });

  it("Save persists the key and closes the dialog", async () => {
    const user = userEvent.setup();
    await openModal();

    await user.type(screen.getByLabelText("API key"), "demo-key-123");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(localStorage.getItem("neurosight.apiKey")).toBe("demo-key-123");
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
  });

  it("Escape closes the dialog without saving", async () => {
    const user = userEvent.setup();
    await openModal();

    await user.type(screen.getByLabelText("API key"), "typed-but-not-saved");
    await user.keyboard("{Escape}");

    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
    expect(localStorage.getItem("neurosight.apiKey")).toBeNull();
  });

  it("Escape closes the dialog even without typing (no key in storage)", async () => {
    const user = userEvent.setup();
    await openModal();

    await user.keyboard("{Escape}");

    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
    expect(localStorage.getItem("neurosight.apiKey")).toBeNull();
  });

  it("Enter with an empty key is a no-op (dialog stays open, nothing saved)", async () => {
    const user = userEvent.setup();
    await openModal();

    await user.keyboard("{Enter}");

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(localStorage.getItem("neurosight.apiKey")).toBeNull();
  });

  it("Enter with whitespace only does not persist a trimmed-empty key", async () => {
    const user = userEvent.setup();
    await openModal();

    await user.type(screen.getByLabelText("API key"), "   ");
    await user.keyboard("{Enter}");

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(localStorage.getItem("neurosight.apiKey")).toBeNull();
  });
});
