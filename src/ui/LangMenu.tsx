import { LANGS, LANG_NAMES, t } from '../i18n/index.ts'
import type { Lang } from '../i18n/index.ts'

interface Props {
  lang: Lang
  onChange: (lang: Lang) => void
}

// Visible language choice. The parent saves the value and uses it on the next visit.
export function LangMenu({ lang, onChange }: Props) {
  return (
    <label className="lang-menu">
      <span>{t(lang, 'lang.label')}</span>
      <select value={lang} onChange={(e) => onChange(e.target.value === 'ko' ? 'ko' : 'en')} aria-label={t(lang, 'lang.label')}>
        {LANGS.map((l) => (
          <option key={l} value={l}>
            {LANG_NAMES[l]}
          </option>
        ))}
      </select>
    </label>
  )
}
