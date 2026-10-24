export type GainsRow = {
  decile: number;
  customers: number;
  adds: number;
  hit_rate: number;
  lift: number;
  cumulative_share_of_adds: number;
};

export type Readout = {
  customers: number; // customers contacted
  adds: number;      // new CASA openers among them
  share: number;     // share of all new openers reached, 0-1
  hitRate: number;   // openers / contacted, 0-1
  lift: number;      // hit rate vs calling customers at random
};

/** What a team that contacts the top `k` deciles (1-10) by model score would get. */
export function readout(gains: GainsRow[], k: number): Readout | null {
  if (gains.length === 0) return null;
  // Cumulative over the top k deciles: a team calls the whole list, not just the k-th slice.
  const top = gains.slice(0, k);
  const customers = top.reduce((s, g) => s + g.customers, 0);
  const adds = top.reduce((s, g) => s + g.adds, 0);
  const allCustomers = gains.reduce((s, g) => s + g.customers, 0);
  const allAdds = gains.reduce((s, g) => s + g.adds, 0);
  const hitRate = adds / customers;
  return { customers, adds, share: adds / allAdds, hitRate, lift: hitRate / (allAdds / allCustomers) };
}
