import type { SVGProps } from 'react'

type IconProps = SVGProps<SVGSVGElement> & { size?: number }

function icon(paths: React.ReactNode, viewBox = '0 0 24 24') {
  return function Icon({ size = 16, ...props }: IconProps) {
    return (
      <svg
        width={size}
        height={size}
        viewBox={viewBox}
        fill="none"
        stroke="currentColor"
        strokeWidth={1.8}
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
        {...props}
      >
        {paths}
      </svg>
    )
  }
}

export const ComposeIcon = icon(
  <>
    <path d="M12 20h9" />
    <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
  </>,
)
export const SearchIcon = icon(
  <>
    <circle cx="11" cy="11" r="7" />
    <path d="m20 20-3.5-3.5" />
  </>,
)
export const FolderIcon = icon(<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z" />)
export const ChevronIcon = icon(<path d="m9 6 6 6-6 6" />)
export const BranchIcon = icon(
  <>
    <circle cx="6" cy="5" r="2" />
    <circle cx="6" cy="19" r="2" />
    <circle cx="18" cy="7" r="2" />
    <path d="M6 7v10" />
    <path d="M18 9v1a4 4 0 0 1-4 4H9a3 3 0 0 0-3 3" />
  </>,
)
export const ArrowUpIcon = icon(
  <>
    <path d="M12 19V5" />
    <path d="m5 12 7-7 7 7" />
  </>,
)
export const ArrowDownIcon = icon(
  <>
    <path d="M12 5v14" />
    <path d="m19 12-7 7-7-7" />
  </>,
)
export const StopIcon = icon(<rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor" stroke="none" />)
export const TerminalIcon = icon(
  <>
    <path d="m5 8 4 4-4 4" />
    <path d="M12 16h7" />
  </>,
)
export const FileIcon = icon(
  <>
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z" />
    <path d="M14 3v5h5" />
  </>,
)
export const PencilIcon = icon(
  <>
    <path d="M4 20h4L18.5 9.5a2.83 2.83 0 0 0-4-4L4 16Z" />
    <path d="m13.5 6.5 4 4" />
  </>,
)
export const ToolIcon = icon(
  <path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.5 2.5-2.4-.6-.6-2.4Z" />,
)
export const BrainIcon = icon(
  <>
    <path d="M12 5a3 3 0 0 0-5.9.8A3 3 0 0 0 4 11a3 3 0 0 0 1.5 4.8A3 3 0 0 0 12 18Z" />
    <path d="M12 5a3 3 0 0 1 5.9.8A3 3 0 0 1 20 11a3 3 0 0 1-1.5 4.8A3 3 0 0 1 12 18Z" />
    <path d="M12 5v13" />
  </>,
)
export const MoreIcon = icon(
  <>
    <circle cx="5" cy="12" r="1.2" fill="currentColor" />
    <circle cx="12" cy="12" r="1.2" fill="currentColor" />
    <circle cx="19" cy="12" r="1.2" fill="currentColor" />
  </>,
)
export const SunIcon = icon(
  <>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </>,
)
export const MoonIcon = icon(<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" />)
export const SidebarIcon = icon(
  <>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <path d="M9 4v16" />
  </>,
)
export const CheckIcon = icon(<path d="m5 12 5 5 9-10" />)
export const QuestionIcon = icon(
  <>
    <circle cx="12" cy="12" r="9" />
    <path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6V14" />
    <path d="M12 17h.01" />
  </>,
)
export const SparkIcon = icon(<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6" />)
export const CodeIcon = icon(
  <>
    <path d="m8 7-5 5 5 5" />
    <path d="m16 7 5 5-5 5" />
  </>,
)
export const BoltIcon = icon(<path d="M13 2 4 14h7l-1 8 9-12h-7Z" />)
export const PaletteIcon = icon(
  <>
    <path d="M12 3a9 9 0 0 0 0 18c1.1 0 1.7-.9 1.3-1.9-.4-1-.1-2.1 1.2-2.1H17a4 4 0 0 0 4-4c0-5.5-4-10-9-10Z" />
    <circle cx="7.5" cy="11" r="1" fill="currentColor" />
    <circle cx="10" cy="7" r="1" fill="currentColor" />
    <circle cx="14.5" cy="7" r="1" fill="currentColor" />
  </>,
)
export const CloseIcon = icon(<path d="M6 6l12 12M18 6 6 18" />)
export const GearIcon = icon(
  <>
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1Z" />
  </>,
)
export const SlidersIcon = icon(
  <path d="M4 7h10M18 7h2M4 17h4M12 17h8M14 4v6M8 14v6" />,
)
export const BackIcon = icon(<path d="m15 6-6 6 6 6" />)
export const PuzzleIcon = icon(
  <path d="M9 4.5a2 2 0 1 1 4 0V6h3a1 1 0 0 1 1 1v3h1.5a2 2 0 1 1 0 4H17v3a1 1 0 0 1-1 1h-3v-1.5a2 2 0 1 0-4 0V18H6a1 1 0 0 1-1-1v-3h1.5a2 2 0 1 0 0-4H5V7a1 1 0 0 1 1-1h3Z" />,
)
export const OrchestratorIcon = icon(
  <>
    <circle cx="12" cy="12" r="2.5" />
    <circle cx="5.5" cy="5.5" r="2" />
    <circle cx="18.5" cy="5.5" r="2" />
    <circle cx="12" cy="20" r="2" />
    <path d="m7 7 3.2 3.2M17 7l-3.2 3.2M12 14.5V18" />
  </>,
)
export const ArchiveIcon = icon(
  <>
    <rect x="3" y="4" width="18" height="4" rx="1" />
    <path d="M5 8v10a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8" />
    <path d="M10 12h4" />
  </>,
)
export const TracesIcon = icon(<path d="M3 12h4l3-8 4 16 3-8h4" />)
export const PanelIcon = icon(
  <>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <path d="M15 4v16" />
  </>,
)
export const ExternalIcon = icon(
  <>
    <path d="M14 4h6v6" />
    <path d="M20 4 11 13" />
    <path d="M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4" />
  </>,
)
export const InfoIcon = icon(
  <>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 11v6M12 7.5h.01" />
  </>,
)
export const KeyIcon = icon(
  <>
    <circle cx="8" cy="15" r="4" />
    <path d="m11 12 9-9M17 6l3 3M14 9l2 2" />
  </>,
)
export const ServerIcon = icon(
  <>
    <rect x="4" y="4" width="16" height="7" rx="2" />
    <rect x="4" y="13" width="16" height="7" rx="2" />
    <path d="M8 7.5h.01M8 16.5h.01" />
  </>,
)
export const PlusIcon = icon(<path d="M12 5v14M5 12h14" />)
export const DiffIcon = icon(
  <>
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z" />
    <path d="M9 10h6M12 7v6M9 17h6" />
  </>,
)
