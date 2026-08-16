/**
 * Copy text to the clipboard, with a fallback for insecure contexts.
 *
 * `navigator.clipboard` only exists in a secure context (HTTPS, or localhost).
 * Serving the UI from a plain-HTTP IP — which is exactly how this app runs on a
 * rented GPU host — leaves it undefined, so the Copy button silently did
 * nothing there. The execCommand path is deprecated but remains the only
 * option available in that context.
 *
 * Returns true on success so callers can show real feedback instead of
 * assuming it worked.
 */
export async function copyText(text: string): Promise<boolean> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Permission denied or blocked — fall through to the legacy path.
    }
  }

  try {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    // Keep it off-screen and non-disruptive: focusing a visible element would
    // scroll the page, and readOnly stops mobile keyboards from opening.
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.top = '-9999px';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(textarea);
    return ok;
  } catch {
    return false;
  }
}
