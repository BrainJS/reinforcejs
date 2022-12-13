import {Mat} from "./mat";

describe("Mat", () => {
  describe("constructor", () => {
    it("sets this values correctly", () => {
      const mat = new Mat(1, 2);
      expect(mat.n).toEqual(1);
      expect(mat.d).toEqual(2);
      expect(mat.w).toEqual(new Float64Array(2));
      expect(mat.dw).toEqual(new Float64Array(2));
    });
  });
  describe("get", () => {
    it("returns value at row & col", () => {
      const mat = new Mat(2, 2);
      mat.w[0] = 1;
      expect(mat.get(0, 0)).toBe(1);
    });
    it("throws if value is outside range", () => {
      const mat = new Mat(2, 2);
      expect(() => mat.get(10, 10)).toThrow();
    });
  });
  describe("set", () => {
    it("sets value at row & col", () => {
      const mat = new Mat(2, 2);
      mat.set(0, 0, 4);
      expect(mat.w[0]).toBe(4);
    });
    it("throws if value is outside range", () => {
      const mat = new Mat(2, 2);
      expect(() => mat.set(10, 10, 4)).toThrow();
    });
  });
  describe("setFrom", () => {
    it("sets value from array", () => {
      const mat = new Mat(2, 2);
      mat.setFrom([1,2,3,4]);
      expect(mat.w).toEqual(new Float64Array([1,2,3,4]));
    });
  });
  describe("setColumn", () => {
    it("sets value from Mat", () => {
      const sourceMat = new Mat(2, 2);
      sourceMat.set(0, 0, 1);
      sourceMat.set(1, 0, 2);
      const targetMat = new Mat(2, 2);
      targetMat.setColumn(sourceMat, 0);
      expect(targetMat.w).toEqual(new Float64Array([1, 0, 0, 0]));
    });
  });
  describe("toJSON", () => {
    it("returns Matrix json", () => {
      const mat = new Mat(2, 2);
      mat.setFrom([1,2,3,4]);
      expect(mat.toJSON()).toEqual({
        n: 2,
        d: 2,
        w: [1, 2, 3, 4],
      });
    });
  });
  describe("fromJSON", () => {
    it("deserializes from json", () => {
      const mat = new Mat(0, 0);
      mat.fromJSON({
        n: 2,
        d: 2,
        w: [1, 2, 3, 4],
      });
      expect(mat.n).toEqual(2);
      expect(mat.d).toEqual(2);
      expect(mat.w).toEqual(new Float64Array([1,2,3,4]));
      expect(mat.dw).toEqual(new Float64Array([0,0,0,0]));
    });
  });
  describe("clone", () => {
    it("creates new Mat, with copy of w", () => {
      const mat = new Mat(2, 2);
      mat.setFrom([1,2,3,4]);
      const clone = mat.clone();
      expect(clone).toEqual(mat);
    });
  });
  describe("update", () => {
    describe("when dw values are not 0", () => {
      it("sets w with alpha and resets dw", () => {
        const mat = new Mat(2, 2);
        mat.setFrom([1,2,3,4]);
        mat.dw = new Float64Array([1,1,1,1]);
        mat.update(4);
        expect(mat.w).toEqual(new Float64Array([-3,-2,-1,0]));
        expect(mat.dw).toEqual(new Float64Array([0,0,0,0]));
      });
    });
  });
  describe("flattenGrad", () => {
    it("sets w from m.dw", () => {
      const sourceMat = new Mat(2, 2);
      sourceMat.dw = new Float64Array([1,2,3,4]);
      const mat = new Mat(2, 2);
      mat.dw = new Float64Array([1,2,3,4]);
      const index = mat.flattenGrad(sourceMat, 0);
      expect(mat.w).toEqual(sourceMat.dw);
      expect(index).toEqual(4);
    });
  });
  describe("gradFillConst", () => {
    it("fills dw with value", () => {
      const mat = new Mat(2, 2);
      mat.gradFillConst(2);
      expect(mat.dw).toEqual(new Float64Array([2,2,2,2]));
    });
  });
  describe("fillRandn", () => {
    it("fills w with random values", () => {
      const mat = new Mat(2, 2);
      mat.fillRandn(1, 2);
      expect(mat.w).not.toEqual(new Float64Array([0,0,0,0]));
    });
  });
  describe("fillRand", () => {
    it("fills w with random values", () => {
      const mat = new Mat(2, 2);
      mat.fillRand(1, 2);
      expect(mat.w).not.toEqual(new Float64Array([0,0,0,0]));
    });
  });
  describe("maxi", () => {
    it("returns index of max value", () => {
      const mat = new Mat(2, 2);
      mat.w[2] = 4;
      expect(mat.maxi()).toEqual(2);
    });
  });
  describe("static fromJSON", () => {
    it("returns a Mat from json", () => {
      const mat = Mat.fromJSON({
        n: 2,
        d: 2,
        w: [1,2,3,4],
      });
      const expectedMat = new Mat(2, 2);
      expectedMat.w = new Float64Array([1,2,3,4]);
      expect(mat).toEqual(expectedMat);
    });
  });
});
