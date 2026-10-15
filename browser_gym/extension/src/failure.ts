/** A place a check failed: a label and a few numbered source lines, the failing one marked. */
export interface FailurePlace {
  label: string;
  lines: Array<{ number: number; text: string; failing: boolean }>;
}

/** The sources a traceback's frames point into, by the file names the checker compiles them under. */
export interface FailureSources {
  /** Each verifier's source, by name: they are compiled one at a time as `checker.py`. */
  verifiers: Record<string, string>;
  /** Helpers shared by the verifiers, compiled as `checker_helpers.py`. */
  helpers?: string;
  /** Shared selected program, compiled with its helpers in source order. */
  program?: string;
  /** The submitted answer, compiled as `solution.py`. */
  submitted?: string;
}

interface Frame {
  file: string;
  line: number;
  name: string;
}

const FRAME = /File "([^"]+)", line (\d+), in ([\w<>]+)/g;

function excerpt(source: string, line: number): FailurePlace['lines'] {
  const lines = source.split('\n');
  const first = Math.max(1, line - 2);
  const last = Math.min(lines.length, line + 1);
  const shown: FailurePlace['lines'] = [];
  for (let number = first; number <= last; number++) {
    shown.push({ number, text: lines[number - 1], failing: number === line });
  }
  return shown;
}

function place(frame: Frame, sources: FailureSources): FailurePlace | undefined {
  if (frame.file === 'solution.py' && sources.submitted) {
    return { label: `In your answer, line ${frame.line} (${frame.name})`, lines: excerpt(sources.submitted, frame.line) };
  }
  if (frame.file === 'checker_program.py' && sources.program) {
    return { label: `In the check ${frame.name}, line ${frame.line}`, lines: excerpt(sources.program, frame.line) };
  }
  if (frame.file === 'checker.py' && sources.verifiers[frame.name]) {
    return { label: `In the check ${frame.name}, line ${frame.line}`, lines: excerpt(sources.verifiers[frame.name], frame.line) };
  }
  if (frame.file === 'checker_helpers.py' && sources.helpers) {
    return { label: `In the check helper ${frame.name}, line ${frame.line}`, lines: excerpt(sources.helpers, frame.line) };
  }
  return undefined;
}

/**
 * Where a failed check failed, from its traceback. When the answer raised,
 * the answer's line comes first, then the check line that called it;
 * otherwise the check line whose assertion failed.
 */
export function failurePlaces(traceback: string, sources: FailureSources): FailurePlace[] {
  const frames: Frame[] = [...traceback.matchAll(FRAME)].map(match => ({ file: match[1], line: Number(match[2]), name: match[3] }));
  const located = frames.map(frame => ({ frame, place: place(frame, sources) }))
    .filter((item): item is { frame: Frame; place: FailurePlace } => Boolean(item.place));
  const inCheck = located.filter(item => item.frame.file !== 'solution.py').pop();
  const innermost = located[located.length - 1];
  if (innermost?.frame.file === 'solution.py') {
    return [innermost.place, ...(inCheck ? [inCheck.place] : [])];
  }
  return inCheck ? [inCheck.place] : [];
}
