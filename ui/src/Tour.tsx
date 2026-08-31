/**
 * A guided walkthrough for someone opening this cold.
 *
 * A judge has a few minutes and no context. Left alone with a dashboard they
 * will click the biggest button, get a graph of coloured dots, and never find
 * the thing the project is actually about. So the tour drives the console
 * itself: each step can select a ring, confirm an account, or start the replay
 * before it explains what just happened. Nothing depends on the visitor
 * clicking the right thing in the right order.
 *
 * Hand-rolled rather than pulled from a library, because the whole thing is a
 * spotlight, a card and an index — and a dependency would still need every
 * step of copy written by hand.
 */

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

export interface TourStep {
  /** Value of the `data-tour` attribute on the element to highlight. */
  target: string;
  title: string;
  body: React.ReactNode;
  /** Where the card sits relative to the target. Flips if there is no room. */
  placement?: "top" | "bottom" | "left" | "right";
  /** Run before the step is shown — this is how the tour drives the app. */
  before?: () => void | Promise<void>;
  /** Extra settle time in ms, for a step whose action fetches. */
  settleMs?: number;
}

interface Rect {
  top: number;
  left: number;
  width: number;
  height: number;
}

const CARD_WIDTH = 380;
const GAP = 14;
const PAD = 8;

function place(
  rect: Rect | null,
  preferred: TourStep["placement"],
  cardHeight: number
): { top: number; left: number; arrow: string } {
  const vw = window.innerWidth;
  const vh = window.innerHeight;

  if (!rect) {
    return { top: vh / 2 - cardHeight / 2, left: vw / 2 - CARD_WIDTH / 2, arrow: "none" };
  }

  const room = {
    top: rect.top,
    bottom: vh - (rect.top + rect.height),
    left: rect.left,
    right: vw - (rect.left + rect.width),
  };

  let side = preferred ?? "bottom";
  // Flip when the preferred side cannot hold the card.
  if (side === "bottom" && room.bottom < cardHeight + GAP) side = "top";
  if (side === "top" && room.top < cardHeight + GAP) side = "bottom";
  if (side === "right" && room.right < CARD_WIDTH + GAP) side = "left";
  if (side === "left" && room.left < CARD_WIDTH + GAP) side = "right";

  let top: number;
  let left: number;
  if (side === "bottom") {
    top = rect.top + rect.height + GAP;
    left = rect.left + rect.width / 2 - CARD_WIDTH / 2;
  } else if (side === "top") {
    top = rect.top - cardHeight - GAP;
    left = rect.left + rect.width / 2 - CARD_WIDTH / 2;
  } else if (side === "right") {
    top = rect.top + rect.height / 2 - cardHeight / 2;
    left = rect.left + rect.width + GAP;
  } else {
    top = rect.top + rect.height / 2 - cardHeight / 2;
    left = rect.left - CARD_WIDTH - GAP;
  }

  // Keep the card on screen whatever the target does.
  left = Math.max(12, Math.min(left, vw - CARD_WIDTH - 12));
  top = Math.max(12, Math.min(top, vh - cardHeight - 12));
  return { top, left, arrow: side };
}

export function Tour({
  steps,
  open,
  onClose,
}: {
  steps: TourStep[];
  open: boolean;
  onClose: () => void;
}) {
  const [index, setIndex] = useState(0);
  const [rect, setRect] = useState<Rect | null>(null);
  const [busy, setBusy] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);
  const [cardHeight, setCardHeight] = useState(220);

  const step = steps[index];

  /**
   * Steps are read through a ref, and the effects below key on the *index*
   * only.
   *
   * A step's `before` drives the app — confirming an account, loading a ring —
   * which changes App's state, which recreates the callbacks the step list is
   * built from, which produces a new `step` object. Keying the effect on that
   * object meant running `before` again, changing state again: an infinite
   * loop that fires the confirm endpoint forever. The index is the only thing
   * that should advance a step.
   */
  const stepsRef = useRef(steps);
  stepsRef.current = steps;

  useEffect(() => {
    if (open) setIndex(0);
  }, [open]);

  /** Run the step's action, then find and measure its target. */
  useEffect(() => {
    if (!open) return;
    const current = stepsRef.current[index];
    if (!current) return;
    let cancelled = false;

    (async () => {
      setBusy(true);
      try {
        await current.before?.();
      } catch {
        // A step that cannot set itself up should not strand the tour.
      }
      await new Promise((r) => setTimeout(r, current.settleMs ?? 60));
      if (cancelled) return;

      const el = document.querySelector<HTMLElement>(`[data-tour="${current.target}"]`);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        // Let the smooth scroll finish before measuring, or the spotlight
        // lands where the element used to be.
        await new Promise((r) => setTimeout(r, 320));
        if (cancelled) return;
        const r = el.getBoundingClientRect();
        setRect({ top: r.top, left: r.left, width: r.width, height: r.height });
      } else {
        setRect(null);
      }
      setBusy(false);
    })();

    return () => {
      cancelled = true;
    };
  }, [open, index]);

  useLayoutEffect(() => {
    if (cardRef.current) setCardHeight(cardRef.current.offsetHeight);
  }, [index, step, rect]);

  // Keep the spotlight on the target when the window changes underneath it.
  useEffect(() => {
    if (!open) return;
    const remeasure = () => {
      const target = stepsRef.current[index]?.target;
      const el = target ? document.querySelector<HTMLElement>(`[data-tour="${target}"]`) : null;
      if (!el) return;
      const r = el.getBoundingClientRect();
      setRect({ top: r.top, left: r.left, width: r.width, height: r.height });
    };
    window.addEventListener("resize", remeasure);
    window.addEventListener("scroll", remeasure, true);
    return () => {
      window.removeEventListener("resize", remeasure);
      window.removeEventListener("scroll", remeasure, true);
    };
  }, [open, index]);

  const next = useCallback(() => {
    setIndex((i) => {
      if (i + 1 >= steps.length) {
        onClose();
        return i;
      }
      return i + 1;
    });
  }, [steps.length, onClose]);

  const back = useCallback(() => setIndex((i) => Math.max(0, i - 1)), []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowRight" || e.key === "Enter") next();
      if (e.key === "ArrowLeft") back();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, next, back, onClose]);

  if (!open || !step) return null;

  const pos = place(rect, step.placement, cardHeight);
  const isLast = index === steps.length - 1;

  return (
    <div className="fixed inset-0 z-50">
      {/*
        The spotlight is one element: a box the size of the target, with a
        shadow large enough to cover the rest of the screen. Cheaper and
        crisper than four dimming panels, and it animates as one thing.
      */}
      {rect ? (
        <div
          className="pointer-events-none absolute rounded-lg ring-2 ring-sky-400/70 transition-all duration-300"
          style={{
            top: rect.top - PAD,
            left: rect.left - PAD,
            width: rect.width + PAD * 2,
            height: rect.height + PAD * 2,
            boxShadow: "0 0 0 9999px rgba(2,6,12,0.78)",
          }}
        />
      ) : (
        <div className="absolute inset-0 bg-[#02060c]/80" />
      )}

      {/* Swallow clicks so the tour stays in control of the sequence. */}
      <div className="absolute inset-0" onClick={(e) => e.stopPropagation()} />

      <div
        ref={cardRef}
        className="absolute rounded-xl border border-slate-700 bg-slate-900 shadow-2xl transition-all duration-300"
        style={{ top: pos.top, left: pos.left, width: CARD_WIDTH }}
      >
        <div className="flex items-center justify-between border-b border-slate-800 px-4 py-2.5">
          <span className="text-[11px] font-medium uppercase tracking-wider text-sky-400">
            Step {index + 1} of {steps.length}
          </span>
          <button
            onClick={onClose}
            className="text-[11px] text-slate-500 transition hover:text-slate-300"
          >
            Skip tour
          </button>
        </div>

        <div className="px-4 py-3">
          <h3 className="text-sm font-semibold text-slate-100">{step.title}</h3>
          <div className="mt-1.5 text-[13px] leading-relaxed text-slate-400">{step.body}</div>
        </div>

        <div className="flex items-center justify-between gap-3 border-t border-slate-800 px-4 py-2.5">
          <div className="flex gap-1">
            {steps.map((_, i) => (
              <span
                key={i}
                className={`h-1 w-4 rounded-full transition ${
                  i === index ? "bg-sky-400" : i < index ? "bg-slate-600" : "bg-slate-800"
                }`}
              />
            ))}
          </div>
          <div className="flex items-center gap-2">
            {index > 0 && (
              <button
                onClick={back}
                className="rounded-md px-3 py-1 text-xs text-slate-400 transition hover:text-slate-200"
              >
                Back
              </button>
            )}
            <button
              onClick={next}
              disabled={busy}
              className="rounded-md bg-sky-500/20 px-3.5 py-1 text-xs font-medium text-sky-200 ring-1 ring-sky-500/40 transition hover:bg-sky-500/30 disabled:opacity-50"
            >
              {busy ? "…" : isLast ? "Finish" : "Next"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
