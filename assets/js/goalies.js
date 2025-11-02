import { getSummaryData, getGoalieMap } from './data.js';
import {
  attachNavHighlight,
  computeUniqueTeams,
  formatNumber,
  formatPercent,
  parseSearchTerms,
  sortData,
} from './utils.js';

const state = {
  team: 'ALL',
  searchTerms: [],
  minShots: 0,
  sortKey: 'final_save_pct',
  ascending: false,
};

let summary = [];
let goalieMap = {};

function populateTeamFilter(teams) {
  const select = document.getElementById('goalie-team-filter');
  if (!select) return;
  teams.forEach((team) => {
    const option = document.createElement('option');
    option.value = team;
    option.textContent = team;
    select.appendChild(option);
  });
}

function updateSortIndicators() {
  const headers = document.querySelectorAll('th.sortable');
  headers.forEach((th) => {
    const { key } = th.dataset;
    if (key === state.sortKey) {
      th.setAttribute('data-sort', state.ascending ? 'asc' : 'desc');
    } else {
      th.removeAttribute('data-sort');
    }
  });
}

function filterRows() {
  return summary.filter((row) => {
    const matchesTeam = state.team === 'ALL' || row.team === state.team;
    const matchesSearch =
      !state.searchTerms.length || state.searchTerms.some((term) => row.goalie.toLowerCase().includes(term));
    const meetsShots = (row.total_shots || 0) >= state.minShots;
    return matchesTeam && matchesSearch && meetsShots;
  });
}

function renderTable(rows) {
  const tbody = document.getElementById('goalie-table-body');
  const empty = document.getElementById('goalie-empty');
  if (!tbody || !empty) return;
  tbody.innerHTML = '';

  if (!rows.length) {
    empty.hidden = false;
    return;
  }

  empty.hidden = true;
  const fragment = document.createDocumentFragment();
  rows.forEach((row) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${row.team || 'Okänt lag'}</td>
      <td>${row.goalie}</td>
      <td>${formatNumber(row.games_played)}</td>
      <td>${formatNumber(row.total_shots)}</td>
      <td>${formatNumber(row.total_saves)}</td>
      <td>${formatNumber(row.goals_against)}</td>
      <td>${row.avg_shots_per_game ? row.avg_shots_per_game.toFixed(1) : '–'}</td>
      <td>${formatPercent(row.final_save_pct, 1)}</td>
    `;
    fragment.appendChild(tr);
  });
  tbody.appendChild(fragment);
}

function updateCounters(total, filtered) {
  const totalEl = document.getElementById('goalie-total');
  const filteredEl = document.getElementById('goalie-filtered');
  if (totalEl) totalEl.textContent = formatNumber(total);
  if (filteredEl) filteredEl.textContent = formatNumber(filtered);
}

function renderHighlights(rows) {
  const highlight = document.getElementById('goalie-highlight');
  if (!highlight) return;
  if (!rows.length) {
    highlight.textContent = 'Inga målvakter matchar nuvarande filter.';
    return;
  }
  const leader = rows[0];
  const leaderTeam = goalieMap[leader.goalie] || leader.team || 'Okänt lag';
  highlight.innerHTML = `
    <strong>${leader.goalie}</strong> (${leaderTeam}) toppar listan med ${formatPercent(
      leader.final_save_pct,
      1,
    )} på ${formatNumber(leader.total_shots)} skott.
  `;
}

function applyStateChange() {
  const rows = sortData(filterRows(), state.sortKey, state.ascending);
  renderTable(rows);
  updateCounters(summary.length, rows.length);
  renderHighlights(rows);
}

function registerEvents() {
  const teamSelect = document.getElementById('goalie-team-filter');
  const searchInput = document.getElementById('goalie-search');
  const shotSelect = document.getElementById('goalie-shot-filter');
  const resetButton = document.getElementById('goalie-reset');

  teamSelect?.addEventListener('change', (event) => {
    state.team = event.target.value;
    applyStateChange();
  });

  searchInput?.addEventListener('input', (event) => {
    state.searchTerms = parseSearchTerms(event.target.value);
    applyStateChange();
  });

  shotSelect?.addEventListener('change', (event) => {
    state.minShots = Number(event.target.value) || 0;
    applyStateChange();
  });

  resetButton?.addEventListener('click', () => {
    state.team = 'ALL';
    state.searchTerms = [];
    state.minShots = 0;
    state.sortKey = 'final_save_pct';
    state.ascending = false;
    if (teamSelect) teamSelect.value = 'ALL';
    if (searchInput) searchInput.value = '';
    if (shotSelect) shotSelect.value = '0';
    updateSortIndicators();
    applyStateChange();
  });

  document.querySelectorAll('th.sortable').forEach((th) => {
    th.addEventListener('click', () => {
      const { key } = th.dataset;
      if (!key) return;
      if (state.sortKey === key) {
        state.ascending = !state.ascending;
      } else {
        state.sortKey = key;
        state.ascending = key === 'goalie' || key === 'team';
      }
      updateSortIndicators();
      applyStateChange();
    });
  });
}

async function init() {
  attachNavHighlight('goalies');
  try {
    [summary, goalieMap] = await Promise.all([getSummaryData(), getGoalieMap()]);
    const teams = computeUniqueTeams(summary);
    populateTeamFilter(teams);
    updateCounters(summary.length, summary.length);
    updateSortIndicators();
    registerEvents();
    applyStateChange();
  } catch (error) {
    console.error('Failed to initialise goalie explorer', error);
    const tableWrapper = document.getElementById('goalie-table');
    if (tableWrapper) {
      tableWrapper.textContent = 'Det gick inte att ladda målvaktslistan.';
    }
  }
}

document.addEventListener('DOMContentLoaded', init);
