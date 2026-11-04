// Who is asking. This server never sees a password or a session cookie: the home
// server authenticates the browser, decides what the person may do, and forwards
// that in headers signed with a shared secret (home's docs/apps-platform-plan.md
// → The contract). Anything that arrives without the right secret is anonymous,
// whatever headers it carries.
//
// The engine keeps no users and no permissions of its own. Only an admin — home's
// own server, scanning in the system identity, or the admin reading the log — has
// anything to ask it.
import type { IncomingHttpHeaders } from "node:http";

export const GATEWAY_HEADER = "x-isputnik-gateway";
export const USER_ID_HEADER = "x-isputnik-user-id";
export const USER_NAME_HEADER = "x-isputnik-user-name";
export const USER_ROLE_HEADER = "x-isputnik-user-role";
/** The browser's address as the gateway saw it, so a log line names the person's machine. */
export const CLIENT_IP_HEADER = "x-isputnik-client-ip";
/** The home server's id for the same request, so its log line and this one match up. */
export const REQUEST_ID_HEADER = "x-isputnik-request-id";

export type Role = "admin" | "member";

export interface Identity {
  id: string;
  name: string;
  /** Home's own role: an admin may call the admin routes. */
  role: Role;
}

export function single(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

function decodeName(raw: string): string {
  try {
    return decodeURIComponent(raw);
  } catch {
    return raw;
  }
}

/**
 * The identity the gateway forwarded, or null when the request is not from the
 * gateway (no secret configured, wrong secret, or no user headers). An empty
 * configured secret never matches anything.
 */
export function identityFrom(headers: IncomingHttpHeaders, secret: string | null): Identity | null {
  if (!secret) return null;
  if (single(headers[GATEWAY_HEADER]) !== secret) return null;
  const id = single(headers[USER_ID_HEADER])?.trim();
  if (!id || id.length > 64) return null;
  const role = single(headers[USER_ROLE_HEADER]);
  if (role !== "admin" && role !== "member") return null;
  const name = decodeName(single(headers[USER_NAME_HEADER]) ?? "").trim().slice(0, 120) || id;
  return { id, name, role };
}

/** Whether this request carries the gateway's secret. */
export function fromGateway(headers: IncomingHttpHeaders, secret: string | null): boolean {
  return Boolean(secret) && single(headers[GATEWAY_HEADER]) === secret;
}

/** Why a request was refused, for the event it leaves behind; never the secret. */
export function rejectionReason(headers: IncomingHttpHeaders, secret: string | null): string {
  if (!secret) return "no gateway secret is configured";
  const sent = single(headers[GATEWAY_HEADER]);
  if (sent === undefined) return "no gateway header";
  if (sent !== secret) return "wrong gateway secret";
  return "gateway headers without a valid identity";
}

const IP = /^[0-9a-fA-F.:]{1,45}$/;

/** The address to log: the browser's as the gateway forwarded it, else the socket's. */
export function clientIpFrom(headers: IncomingHttpHeaders, secret: string | null, socketAddress: string | undefined): string {
  if (fromGateway(headers, secret)) {
    const forwarded = single(headers[CLIENT_IP_HEADER])?.trim();
    if (forwarded && IP.test(forwarded)) return forwarded;
  }
  return socketAddress ?? "";
}

/** The gateway's request id for its own requests, so the two servers' lines match. */
export function forwardedRequestId(headers: IncomingHttpHeaders, secret: string | null): string | undefined {
  if (!fromGateway(headers, secret)) return undefined;
  const id = single(headers[REQUEST_ID_HEADER])?.trim();
  return id && /^[A-Za-z0-9._:-]{1,64}$/.test(id) ? id : undefined;
}

/** The shortest secret the image accepts; home's pairing makes 64. */
export const SECRET_MIN_LENGTH = 32;

/** Why the configured secret is not acceptable, or null. Production only. */
export function checkSecret(secret: string | null, production: boolean): string | null {
  if (!production) return null;
  if (!secret) {
    return "GATEWAY_SECRET is not set. The face engine only answers requests signed by isputnik.home with this secret: pair it on Control panel → Settings › Apps.";
  }
  if (/change-me/i.test(secret)) {
    return "GATEWAY_SECRET is still the example placeholder. Pair the engine on Control panel → Settings › Apps and paste the secret it shows.";
  }
  if (secret.length < SECRET_MIN_LENGTH) {
    return `GATEWAY_SECRET is too short (${secret.length} characters): use at least ${SECRET_MIN_LENGTH}, as Control panel → Settings › Apps makes.`;
  }
  return null;
}

/** The developer identity a source checkout runs with when no secret is set: an admin. "id:name". */
export function devIdentity(spec: string): Identity {
  const [id = "dev", name = "Developer"] = spec.split(":");
  return { id, name, role: "admin" };
}
