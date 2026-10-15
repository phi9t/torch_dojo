import { describe, expect, it } from 'vitest';
import { paneForCell, type PaneCell } from './cellPanes';

function markdown(source: string, role?: string): PaneCell {
  return { cellType: 'markdown', source, role };
}

function code(source: string, role?: string): PaneCell {
  return { cellType: 'code', source, role };
}

describe('paneForCell', () => {
  it('places generated cells by their dojo role', () => {
    expect(paneForCell(code('def f(): pass', 'solution'))).toBe('answer');
    expect(paneForCell(code('exec(await pyfetch(url))', 'exploration'))).toBe('explore');
    expect(paneForCell(code('probe()', 'drill'))).toBe('drill');
    expect(paneForCell(code('x = 1', 'scratch'))).toBe('scratch');
  });

  it('treats a code cell without a role as scratch', () => {
    expect(paneForCell(code('# Try a small example here.\n'))).toBe('scratch');
  });

  it('places generated instruction cells by their opening line', () => {
    expect(paneForCell(markdown('# 0059 · Stable Softmax\n\n<p>Brief</p>', 'instruction'))).toBe('brief');
    expect(paneForCell(markdown('### Explore\nRun the short cell below.', 'instruction'))).toBe('explore');
    expect(paneForCell(markdown('Edit the solution below. Only this marked cell is sent to Check.', 'instruction')))
      .toBe('answer-note');
    expect(paneForCell(markdown('Scratch space: try your own example.', 'instruction'))).toBe('scratch');
  });

  it('places instruction cells from reps saved before roles were written', () => {
    expect(paneForCell(markdown('# MLP Modeling Lab\n\nInitialize, train'))).toBe('brief');
    expect(paneForCell(markdown('### Explore\nRun the short cell below to see the MLP visuals.'))).toBe('explore');
    expect(paneForCell(markdown('Scratch space: notebook variables here are useful'))).toBe('scratch');
  });

  it('keeps any other markdown with the brief', () => {
    expect(paneForCell(markdown('A note the author added.'))).toBe('brief');
  });
});
