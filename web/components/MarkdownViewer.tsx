'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Props {
  content: string;
}

/**
 * Renders a generated markdown artifact.
 *
 * Transcripts use `## Speaker N` as a turn marker, so h2 is styled as a
 * speaker label (small, indigo, uppercase-ish weight) rather than a section
 * heading — that is what makes a wall of dialogue scannable. Tables get
 * explicit borders because the intelligence artifacts (questions, timeline)
 * lean on them heavily, and they are wrapped so a wide table scrolls itself
 * instead of forcing the whole page sideways on mobile.
 */
export function MarkdownViewer({ content }: Props) {
  return (
    <div
      className="prose prose-slate prose-sm max-w-none
        prose-headings:font-semibold
        prose-h1:mb-4 prose-h1:text-xl prose-h1:text-slate-900
        prose-h2:mb-1 prose-h2:mt-6 prose-h2:text-sm prose-h2:font-semibold prose-h2:tracking-wide prose-h2:text-indigo-600
        prose-h3:text-base prose-h3:text-slate-800
        prose-p:leading-relaxed prose-p:text-slate-700
        prose-strong:text-slate-900
        prose-a:text-indigo-600 prose-a:underline-offset-2
        prose-li:text-slate-700 prose-li:marker:text-slate-400
        prose-table:text-sm
        prose-th:border prose-th:border-slate-200 prose-th:bg-slate-50 prose-th:px-3 prose-th:py-2 prose-th:text-left prose-th:font-semibold prose-th:text-slate-700
        prose-td:border prose-td:border-slate-200 prose-td:px-3 prose-td:py-2 prose-td:align-top
        prose-code:rounded prose-code:bg-slate-100 prose-code:px-1 prose-code:py-0.5 prose-code:text-[0.85em] prose-code:font-normal prose-code:before:content-none prose-code:after:content-none
        prose-blockquote:border-l-indigo-200 prose-blockquote:text-slate-600
        prose-hr:border-slate-200"
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children }) => (
            <div className="my-4 overflow-x-auto rounded-lg border border-slate-200">
              <table className="my-0 w-full border-collapse">{children}</table>
            </div>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
