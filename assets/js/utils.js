export function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '–';
  }
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatNumber(value, options = {}) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '–';
  }
  const formatter = new Intl.NumberFormat('sv-SE', {
    maximumFractionDigits: options.maximumFractionDigits ?? 0,
    minimumFractionDigits: options.minimumFractionDigits ?? 0,
  });
  return formatter.format(value);
}

export function formatDecimal(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '–';
  }
  return Number(value).toFixed(digits);
}

export function computeUniqueTeams(summary) {
  return [...new Set(summary.map((row) => row.team).filter(Boolean))].sort((a, b) => a.localeCompare(b));
}

export function computeZeroShotGoalies(summary) {
  return summary.filter((row) => !row.total_shots);
}

export function computeDateRange(timelineData) {
  const allDates = timelineData
    .flatMap((trace) => trace.x)
    .filter(Boolean)
    .map((value) => new Date(value))
    .filter((d) => !Number.isNaN(d.valueOf()));
  if (!allDates.length) {
    return null;
  }
  const sorted = allDates.sort((a, b) => a - b);
  return { start: sorted[0], end: sorted[sorted.length - 1] };
}

export function createElement(tag, options = {}) {
  const el = document.createElement(tag);
  if (options.className) {
    el.className = options.className;
  }
  if (options.text) {
    el.textContent = options.text;
  }
  if (options.html) {
    el.innerHTML = options.html;
  }
  return el;
}

export function createOption(value, label = value) {
  const option = document.createElement('option');
  option.value = value;
  option.textContent = label;
  return option;
}

export function toId(str) {
  return str.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
}

export function attachNavHighlight(currentPage) {
  const links = document.querySelectorAll('.nav-links a');
  links.forEach((link) => {
    if (link.dataset.page === currentPage) {
      link.setAttribute('aria-current', 'page');
    } else {
      link.removeAttribute('aria-current');
    }
  });
}

export function sortData(rows, key, ascending = true) {
  const factor = ascending ? 1 : -1;
  return [...rows].sort((a, b) => {
    const av = a[key];
    const bv = b[key];
    if (av === null || av === undefined) return 1 * factor;
    if (bv === null || bv === undefined) return -1 * factor;
    if (typeof av === 'number' && typeof bv === 'number') {
      return (av - bv) * factor;
    }
    return String(av).localeCompare(String(bv)) * factor;
  });
}
