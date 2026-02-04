# EXAIM Next.js Frontend (Shadcn UI + Zinc Theme)

Modern Next.js 14 (App Router) implementation of the EXAIM Clinical Decision Support System frontend with Shadcn UI components, zinc dark theme, and real-time token-by-token WebSocket streaming.

## Features

- ✅ **Real-time Token Streaming**: Token-by-token streaming (no batching) for immediate updates
- ✅ **Shadcn UI Components**: Modern, accessible UI components with zinc dark theme
- ✅ **Smart Auto-Scroll**: Maintains scroll position with 50px threshold detection
- ✅ **Summary Accordion**: Only one expanded summary at a time for cognitive load management
- ✅ **WebSocket Resilience**: Automatic reconnection with fixed 3s delay (max 5 attempts)
- ✅ **Accessible Modal**: Focus lock, ESC key, backdrop click, ARIA labels
- ✅ **Responsive Design**: Mobile-first with breakpoint at 768px
- ✅ **Performance Optimized**: React.memo, Zustand selectors

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS with zinc dark theme
- **UI Components**: Shadcn UI
- **State Management**: Zustand
- **Animations**: Framer Motion
- **Accessibility**: Radix UI primitives (via Shadcn)

## Quick Start

### Prerequisites

- Node.js 18+
- FastAPI backend running on `localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Environment Variables

Create a `.env.local` file in the `frontend` directory with the following variables:

```env
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Note:** The `.env.local` file is excluded from version control (via `.gitignore`) to protect sensitive configuration.

**WebSocket URL Validation:** The application validates that `NEXT_PUBLIC_WS_URL` starts with `ws://` or `wss://` on startup. An invalid URL will cause a clear error message.

## Project Structure

```
├── app/
│   ├── layout.tsx              # Root layout with dark mode and modal portal
│   ├── page.tsx                # Main page with WebSocket init
│   └── globals.css             # Global Tailwind styles with zinc theme
├── components/
│   ├── ui/                     # Shadcn UI components
│   ├── Header.tsx              # Connection status header with Badge
│   ├── CaseInput.tsx           # Auto-resize textarea form with Shadcn components
│   ├── AgentWindow.tsx         # Memoized agent trace window with Card
│   ├── AgentWindowContent.tsx  # Auto-scroll implementation
│   ├── AgentTracesPanel.tsx    # Left panel container with Card
│   ├── SummaryCard.tsx         # Memoized summary card with Accordion
│   ├── SummariesPanel.tsx      # Right panel container with Card
│   └── AgentModal.tsx          # Dialog-based modal
├── lib/
│   ├── types.ts                # TypeScript interfaces
│   ├── utils.ts                # Shadcn utility functions
│   └── websocket.ts            # Singleton WebSocket service with URL validation
├── store/
│   └── cdssStore.ts            # Zustand global state with token-by-token streaming
├── next.config.ts              # Next.js config
└── components.json             # Shadcn configuration
```

## Key Implementation Details

### Token Streaming (`store/cdssStore.ts`)
- **Token-by-token updates**: Tokens are immediately appended to `fullText` when received
- **No batching**: Removed 50ms batching interval for instant updates
- **Direct state updates**: Each token triggers an immediate Zustand state update

### WebSocket Service (`lib/websocket.ts`)
- Singleton pattern prevents duplicate connections
- Fixed 3000ms reconnection delay (NOT exponential backoff)
- Max 5 reconnection attempts
- Connection state safeguards

### State Management (`store/cdssStore.ts`)
- Zustand store with `subscribeWithSelector` middleware
- Selector hooks prevent cross-agent re-renders
- Global reset on `processing_started` message

### Auto-Scroll (`components/AgentWindowContent.tsx`)
- 50px threshold for "at bottom" detection
- `requestAnimationFrame` for smooth scrolling
- Respects user manual scrolling

### Shadcn UI Components
- **Card**: Used for panels and agent windows
- **Badge**: Connection status indicator
- **Button**: Form submission and actions
- **Textarea**: Case input with auto-resize
- **Dialog**: Modal for full agent trace view
- **Accordion**: Summary cards with single-expand behavior

## Development Commands

```bash
npm run dev      # Development server
npm run build    # Production build
npm run start    # Start production server
npm run lint     # Lint check
```

## Differences from frontend-old

1. **Token Streaming**: Fixed to stream token-by-token instead of batched updates
2. **UI Components**: Migrated to Shadcn UI components for better accessibility and consistency
3. **Theme**: Zinc dark theme applied throughout
4. **Styling**: Uses CSS variables and theme tokens instead of hardcoded colors

## Troubleshooting

### WebSocket Won't Connect
- Verify FastAPI running on port 8000
- Check CORS middleware added to backend
- Confirm `.env.local` has correct URLs

### Duplicate Connections in Dev Tools
- `reactStrictMode: false` in `next.config.ts` is intentional
- Dev-only behavior; production won't have this issue

## License

Same as parent EXAIM project.
