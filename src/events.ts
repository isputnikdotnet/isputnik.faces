// The record of what happened, for the admin to read from the home app without a
// shell on the host (Control panel → Overview › Logs). Not the request log: that
// is pino on stdout, with the last lines kept in memory (log-buffer.ts). An event
// is something a person would ask about — a refused caller, the models loading,
// a photo that would not decode, a crash.
//
// Games and video keep theirs in a table; this engine has no database, so the
// events live in memory too and start over with the process. The same shape
// comes back over the wire, so home's Logs page reads them as it reads the others.

export type EventLevel = "info" | "warn" | "error";
export const EVENT_LEVELS: readonly EventLevel[] = ["info", "warn", "error"];

export interface EventInput {
  level: EventLevel;
  /** Dotted, category first: "gateway.rejected", "engine.loaded", "detect.failed". */
  event: string;
  actorId?: string | null;
  actorName?: string | null;
  ip?: string | null;
  requestId?: string | null;
  detail?: string;
}

export interface EventRow {
  id: number;
  at: string;
  level: EventLevel;
  event: string;
  actorId: string | null;
  actorName: string | null;
  libraryId: null;
  ip: string | null;
  requestId: string | null;
  detail: string;
}

/** How many are kept: the newest ones, enough for a night's scan to be read back. */
export const EVENT_CAPACITY = 2000;
/** What home shows as the retention: the container's lifetime, in days, is unknown — 0 says "until restart". */
export const EVENT_RETENTION_DAYS = 0;

/** How long a detail may run: a message, not a dump. */
const DETAIL_MAX = 500;

export interface EventQuery {
  level?: EventLevel | null;
  /** A category ("detect") lists everything under it; a full name ("detect.failed") exactly that. */
  event?: string | null;
  page: number;
  pageSize: number;
}

export interface EventPage {
  events: EventRow[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
  /** Every event name on record, for a filter list. */
  facets: { event: string[] };
}

export class EventLog {
  private rows: EventRow[] = [];
  private nextId = 1;

  constructor(private readonly capacity = EVENT_CAPACITY) {}

  record(input: EventInput): void {
    this.rows.push({
      id: this.nextId++,
      at: new Date().toISOString(),
      level: input.level,
      event: input.event,
      actorId: input.actorId ?? null,
      actorName: input.actorName ?? null,
      libraryId: null,
      ip: input.ip ?? null,
      requestId: input.requestId ?? null,
      detail: (input.detail ?? "").slice(0, DETAIL_MAX)
    });
    if (this.rows.length > this.capacity) this.rows.splice(0, this.rows.length - this.capacity);
  }

  /** Newest first, like the others. */
  list(query: EventQuery): EventPage {
    const matches = this.rows.filter((row) => {
      if (query.level && row.level !== query.level) return false;
      if (query.event && row.event !== query.event && !row.event.startsWith(`${query.event}.`)) return false;
      return true;
    }).reverse();
    const total = matches.length;
    const totalPages = Math.max(1, Math.ceil(total / query.pageSize));
    const page = Math.min(query.page, totalPages);
    return {
      events: matches.slice((page - 1) * query.pageSize, page * query.pageSize),
      total,
      page,
      pageSize: query.pageSize,
      totalPages,
      facets: { event: [...new Set(this.rows.map((row) => row.event))].sort() }
    };
  }

  /** How many of one event in the last `hours`. */
  countSince(event: string, hours: number): number {
    const since = Date.now() - hours * 3_600_000;
    return this.rows.filter((row) => row.event === event && Date.parse(row.at) >= since).length;
  }
}
