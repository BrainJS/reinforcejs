// return Mat but filled with random numbers from gaussian
import { Mat } from "./mat";

export class RandMat extends Mat {
  constructor(n: number, d: number, mu: number, std: number) {
    super(n, d);
    this.fillRandn(mu, std);
    //fillRand(this,-std,std); // kind of :P
  }
}
