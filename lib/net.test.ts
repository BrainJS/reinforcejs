import { Net } from "./net";
import { RandMat } from "./rand-mat";
import { Mat } from "./mat";

describe("Net", () => {
  describe("constructor", () => {
    it("sets props correctly", () => {
      const net = new Net(6, 5, 7);
      expect(net.W1 instanceof RandMat).toBeTruthy();
      expect(net.b1 instanceof Mat).toBeTruthy();
      expect(net.W2 instanceof RandMat).toBeTruthy();
      expect(net.b2 instanceof Mat).toBeTruthy();
    });
  });
  describe("update", () => {
    it("calls update(alpha) on W1, b1, W2, b2", () => {
      const net = new Net(1, 2, 3);
      const w1Update = jest.spyOn(net.W1, "update");
      const b1Update = jest.spyOn(net.b1, "update");
      const w2Update = jest.spyOn(net.W2, "update");
      const b2Update = jest.spyOn(net.b2, "update");
      const alpha = .55;
      net.update(alpha);
      expect(w1Update).toHaveBeenCalledWith(alpha);
      expect(b1Update).toHaveBeenCalledWith(alpha);
      expect(w2Update).toHaveBeenCalledWith(alpha);
      expect(b2Update).toHaveBeenCalledWith(alpha);
    });
  });
  describe("zeroGrads", () => {
    it("calls gradFillConst(0) on W1, b1, W2, b2", () => {
      const net = new Net(1, 2, 3);
      const w1GradFillConst = jest.spyOn(net.W1, "gradFillConst");
      const b1GradFillConst = jest.spyOn(net.b1, "gradFillConst");
      const w2GradFillConst = jest.spyOn(net.W2, "gradFillConst");
      const b2GradFillConst = jest.spyOn(net.b2, "gradFillConst");
      net.zeroGrads();
      expect(w1GradFillConst).toHaveBeenCalledWith(0);
      expect(b1GradFillConst).toHaveBeenCalledWith(0);
      expect(w2GradFillConst).toHaveBeenCalledWith(0);
      expect(b2GradFillConst).toHaveBeenCalledWith(0);
    });
  });
  describe("toJSON", () => {
    it("calls toJSON() and returns json from W1, b1, W2, b2", () => {
      const net = new Net(1, 2, 3);
      expect(net.toJSON()).toEqual({
        W1: net.W1.toJSON(),
        b1: net.b1.toJSON(),
        W2: net.W2.toJSON(),
        b2: net.b2.toJSON(),
      });
    });
  });
  describe("fromJSON", () => {
    it("calls fromJSON() provides equivalent net", () => {
      const net = new Net(1, 2 ,3);
      const json = net.toJSON();
      expect(Net.fromJSON(json)).toEqual(net);
    });
  });
});
