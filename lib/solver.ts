import { Mat } from "./mat";

export class Solver {
  decayRate: number;
  smoothEps: number;
  stepCache: { [key: string]: Mat };

  constructor() {
    this.decayRate = 0.999;
    this.smoothEps = 1e-8;
    this.stepCache = {};
  }
  step(model: { [key: string]: Mat }, stepSize: number, regc: number, clipval: number) {
    // perform parameter update
    const solverStats = { ratioClipped: 0 };
    let numClipped = 0;
    let numTot = 0;
    for (const k in model) {
      if (model.hasOwnProperty(k)) {
        const m = model[k]; // mat ref
        if (!(k in this.stepCache)) { this.stepCache[k] = new Mat(m.n, m.d); }
        const s = this.stepCache[k];
        for (let i = 0, n = m.w.length; i < n; i++) {

          // rmsprop adaptive learning rate
          let mdwi = m.dw[i];
          s.w[i] = s.w[i] * this.decayRate + (1 - this.decayRate) * mdwi * mdwi;

          // gradient clip
          if (mdwi > clipval) {
            mdwi = clipval;
            numClipped++;
          }
          if (mdwi < -clipval) {
            mdwi = -clipval;
            numClipped++;
          }
          numTot++;

          // update (and regularize)
          m.w[i] += - stepSize * mdwi / Math.sqrt(s.w[i] + this.smoothEps) - regc * m.w[i];
          m.dw[i] = 0; // reset gradients for next iteration
        }
      }
    }
    solverStats['ratioClipped'] = numClipped * 1 / numTot;
    return solverStats;
  }
}
