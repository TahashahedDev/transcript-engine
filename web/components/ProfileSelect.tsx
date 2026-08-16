'use client';

const PROFILE_DESCRIPTIONS: Record<string, string> = {
  generic: 'General-purpose — works for any meeting',
  banking: 'Financial services — loan terms, banking vocabulary',
  engineering: 'Software engineering — tech terminology',
  healthcare: 'Medical — clinical vocabulary',
  legal: 'Legal — contracts, proceedings',
};

interface Props {
  profiles: string[];
  value: string;
  onChange: (v: string) => void;
}

export function ProfileSelect({ profiles, value, onChange }: Props) {
  return (
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-2">Profile</label>
      <select
        className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-slate-800 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {profiles.map((p) => (
          <option key={p} value={p}>
            {p.charAt(0).toUpperCase() + p.slice(1)}
          </option>
        ))}
      </select>
      {PROFILE_DESCRIPTIONS[value] && (
        <p className="mt-1.5 text-xs text-slate-500">{PROFILE_DESCRIPTIONS[value]}</p>
      )}
    </div>
  );
}
