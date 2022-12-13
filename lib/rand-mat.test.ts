import {RandMat} from "./rand-mat";

describe("RandMat", () => {
  describe("constructor", () => {
    it("sets props and is filled with random values", () => {
      const mat = new RandMat(2, 2, 2, 0.08);
      expect(mat.n).toEqual(2);
      expect(mat.d).toEqual(2);
      expect(mat.mu).toEqual(2);
      expect(mat.std).toEqual(0.08);
      expect(mat.w).not.toEqual(new Float64Array([0,0,0,0]));
    });
  });
});
