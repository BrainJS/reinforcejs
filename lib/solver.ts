import { Mat } from "./mat";

export class Solver {
  decay_rate: number;
  smooth_eps: number;
  step_cache: { [key: string]: Mat };

  constructor() {
    this.decay_rate = 0.999;
    this.smooth_eps = 1e-8;
    this.step_cache = {};
  }
  step(model: { [key: string]: Mat }, step_size: number, regc: number, clipval: number) {
    // perform parameter update
    const solver_stats = { ratio_clipped: 0 };
    let num_clipped = 0;
    let num_tot = 0;
    for (const k in model) {
      if (model.hasOwnProperty(k)) {
        const m = model[k]; // mat ref
        if (!(k in this.step_cache)) { this.step_cache[k] = new Mat(m.n, m.d); }
        const s = this.step_cache[k];
        for(let i = 0, n = m.w.length; i < n; i++) {

          // rmsprop adaptive learning rate
          let mdwi = m.dw[i];
          s.w[i] = s.w[i] * this.decay_rate + (1.0 - this.decay_rate) * mdwi * mdwi;

          // gradient clip
          if (mdwi > clipval) {
            mdwi = clipval;
            num_clipped++;
          }
          if (mdwi < -clipval) {
            mdwi = -clipval;
            num_clipped++;
          }
          num_tot++;

          // update (and regularize)
          m.w[i] += - step_size * mdwi / Math.sqrt(s.w[i] + this.smooth_eps) - regc * m.w[i];
          m.dw[i] = 0; // reset gradients for next iteration
        }
      }
    }
    solver_stats['ratio_clipped'] = num_clipped * 1.0 / num_tot;
    return solver_stats;
  }
}
