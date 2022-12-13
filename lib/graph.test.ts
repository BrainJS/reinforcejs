import {Graph, sig} from "./graph";
import {Mat} from "./mat";

describe("Graph", () => {
  describe("constructor", () => {
    it("defaults needsBackprop to true & sets backprop to []", () => {
      const graph = new Graph();
      expect(graph.needsBackprop).toEqual(true);
      expect(graph.backprop).toEqual([]);
    });
  });
  describe("backward", () => {
    it("iterates backwards, calling backprop[i]", () => {
      const graph = new Graph();
      const array: number[] = [];
      graph.backprop.push(() => array.push(1));
      graph.backprop.push(() => array.push(2));
      graph.backprop.push(() => array.push(3));
      graph.backward();
      expect(array).toEqual([3,2,1]);
    });
  });
  describe("rowPluck", () => {
    it("throws when Mat is out of bounds", () => {
      expect(() => new Graph().rowPluck(new Mat(1, 1), 4)).toThrow();
    });
    it("asserts size", () => {
      const graph = new Graph();
      const m = new Mat(2, 2);
      m.w[0] = 2;
      m.w[1] = 3;
      m.w[2] = 4;
      m.w[3] = 5;
      const expectedM = new Mat(2, 1);
      expectedM.w[0] = 2;
      expectedM.w[1] = 3;
      expect(graph.rowPluck(m, 0)).toEqual(expectedM);
    });
    describe("when used with needsBackprop = false", () => {
      it("does not push anything to this.backprop", () => {
        const graph = new Graph(false);
        graph.rowPluck(new Mat(1, 1), 0);
        expect(graph.backprop.length).toBe(0);
      });
    });
    describe("when used with needsBackprop = true", () => {
      describe("pushed function", () => {
        it("plucks out.dw onto m.dw", () => {
          const graph = new Graph(true);
          const m = new Mat(2, 2);
          m.w[0] = 2;
          m.w[1] = 3;
          m.w[2] = 4;
          m.w[3] = 5;
          const out = graph.rowPluck(m, 0);
          expect(graph.backprop.length).toBe(1);
          out.dw[0] = 2;
          out.dw[1] = 3;
          graph.backprop[0]();
          const expectedM = Mat.fromJSON(m.toJSON());
          expectedM.dw[0] = 2;
          expectedM.dw[1] = 3;
          expectedM.dw[2] = 0;
          expectedM.dw[3] = 0;
          expect(m).toEqual(expectedM);
        });
      });
    });
  });
  describe("tanh", () => {
    it("returns Mat with Mat.w affected by Math.tanh", () => {
      const graph = new Graph();
      expect(graph.tanh(Mat.fromJSON({
        n: 1,
        d: 1,
        w: [1],
      }))).toEqual(Mat.fromJSON({
        n: 1,
        d: 1,
        w: [Math.tanh(1)],
      }));
    });
    describe("when used with needsBackprop = false", () => {
      it("does not push anything to this.backprop", () => {
        const graph = new Graph(false);
        graph.tanh(new Mat(1, 1));
        expect(graph.backprop.length).toBe(0);
      });
    });
    describe("when used with needsBackprop = true", () => {
      describe("pushed function", () => {
        it("backprops tanh out.dw into m.dw", () => {
          const graph = new Graph(true);
          const m = new Mat(2, 2);
          m.w[0] = 2;
          m.w[1] = 3;
          m.w[2] = 4;
          m.w[3] = 5;
          const out = graph.tanh(m);
          expect(graph.backprop.length).toBe(1);
          out.dw[0] = 2;
          out.dw[1] = 3;
          out.dw[2] = 4;
          out.dw[3] = 5;
          graph.backprop[0]();
          const expectedM = Mat.fromJSON(m.toJSON());
          expectedM.dw[0] = 0.14130164970632886;
          expectedM.dw[1] = 0.029598111496320634;
          expectedM.dw[2] = 0.005363802732103462;
          expectedM.dw[3] = 0.0009079161547193015;
          expect(m).toEqual(expectedM);
        });
      });
    });
  });
  describe("sigmoid", () => {
    it("returns Mat with Mat.w affected by sig", () => {
      const graph = new Graph();
      expect(graph.sigmoid(Mat.fromJSON({
        n: 1,
        d: 1,
        w: [1],
      }))).toEqual(Mat.fromJSON({
        n: 1,
        d: 1,
        w: [sig(1)],
      }));
    });
    describe("when used with needsBackprop = false", () => {
      it("does not push anything to this.backprop", () => {
        const graph = new Graph(false);
        graph.sigmoid(new Mat(1, 1));
        expect(graph.backprop.length).toBe(0);
      });
    });
    describe("when used with needsBackprop = true", () => {
      describe("pushed function", () => {
        it("backprops sigmoid out.dw into m.dw", () => {
          const graph = new Graph(true);
          const m = new Mat(2, 2);
          m.w[0] = 2;
          m.w[1] = 3;
          m.w[2] = 4;
          m.w[3] = 5;
          const out = graph.sigmoid(m);
          expect(graph.backprop.length).toBe(1);
          out.dw[0] = 2;
          out.dw[1] = 3;
          out.dw[2] = 4;
          out.dw[3] = 5;
          graph.backprop[0]();
          const expectedM = Mat.fromJSON(m.toJSON());
          expectedM.dw[0] = 0.20998717080701323;
          expectedM.dw[1] = 0.135529979192736;
          expectedM.dw[2] = 0.07065082485316443;
          expectedM.dw[3] = 0.03324028335395016;
          expect(m).toEqual(expectedM);
        });
      });
    });
  });
  describe("relu", () => {
    it("returns Mat with Mat.w affected by relu operation", () => {
      const graph = new Graph();
      expect(graph.relu(Mat.fromJSON({
        n: 1,
        d: 1,
        w: [1],
      }))).toEqual(Mat.fromJSON({
        n: 1,
        d: 1,
        w: [1],
      }));
    });
    describe("when used with needsBackprop = false", () => {
      it("does not push anything to this.backprop", () => {
        const graph = new Graph(false);
        graph.relu(new Mat(1, 1));
        expect(graph.backprop.length).toBe(0);
      });
    });
    describe("when used with needsBackprop = true", () => {
      describe("pushed function", () => {
        it("backprops relu out.dw into m.dw", () => {
          const graph = new Graph(true);
          const m = new Mat(2, 2);
          m.w[0] = 2;
          m.w[1] = 3;
          m.w[2] = 4;
          m.w[3] = 5;
          const out = graph.relu(m);
          expect(graph.backprop.length).toBe(1);
          out.dw[0] = 2;
          out.dw[1] = 3;
          out.dw[2] = 4;
          out.dw[3] = 5;
          graph.backprop[0]();
          const expectedM = Mat.fromJSON(m.toJSON());
          expectedM.dw[0] = 2;
          expectedM.dw[1] = 3;
          expectedM.dw[2] = 4;
          expectedM.dw[3] = 5;
          expect(m).toEqual(expectedM);
        });
      });
    });
  });
  describe("mul", () => {
    it("returns multiplied product of matrices", () => {
      const graph = new Graph();
      expect(graph.mul(Mat.fromJSON({
        n: 1,
        d: 1,
        w: [2],
      }), Mat.fromJSON({
        n: 1,
        d: 1,
        w: [2]
      }))).toEqual(Mat.fromJSON({
        n: 1,
        d: 1,
        w: [4],
      }));
    });
    describe("when used with needsBackprop = false", () => {
      it("does not push anything to this.backprop", () => {
        const graph = new Graph(false);
        graph.mul(new Mat(1, 1), new Mat(1, 1));
        expect(graph.backprop.length).toBe(0);
      });
    });
    describe("when used with needsBackprop = true", () => {
      describe("pushed function", () => {
        it("multiplies out.dw into m1.dw and m2.dw", () => {
          const graph = new Graph(true);
          const m1 = new Mat(2, 2);
          m1.w[0] = 2;
          m1.w[1] = 3;
          m1.w[2] = 4;
          m1.w[3] = 5;
          const m2 = new Mat(2, 2);
          m2.w[0] = 2;
          m2.w[1] = 3;
          m2.w[2] = 4;
          m2.w[3] = 5;
          const out = graph.mul(m1, m2);
          expect(graph.backprop.length).toBe(1);
          out.dw[0] = 2;
          out.dw[1] = 3;
          out.dw[2] = 4;
          out.dw[3] = 5;
          graph.backprop[0]();
          const expectedM1 = Mat.fromJSON(m1.toJSON());
          expectedM1.dw[0] = 13;
          expectedM1.dw[1] = 23;
          expectedM1.dw[2] = 23;
          expectedM1.dw[3] = 41;
          expect(m1).toEqual(expectedM1);
          const expectedM2 = Mat.fromJSON(m2.toJSON());
          expectedM2.dw[0] = 20;
          expectedM2.dw[1] = 26;
          expectedM2.dw[2] = 26;
          expectedM2.dw[3] = 34;
          expect(m2).toEqual(expectedM2);
        });
      });
    });
  });
  describe("add", () => {
    it("returns added product of matrices", () => {
      const graph = new Graph();
      expect(graph.add(Mat.fromJSON({
        n: 1,
        d: 1,
        w: [2],
      }), Mat.fromJSON({
        n: 1,
        d: 1,
        w: [2]
      }))).toEqual(Mat.fromJSON({
        n: 1,
        d: 1,
        w: [4],
      }));
    });
    describe("when used with needsBackprop = false", () => {
      it("does not push anything to this.backprop", () => {
        const graph = new Graph(false);
        graph.add(new Mat(1, 1), new Mat(1, 1));
        expect(graph.backprop.length).toBe(0);
      });
    });
    describe("when used with needsBackprop = true", () => {
      describe("pushed function", () => {
        it("added out.dw into m1.dw and m2.dw", () => {
          const graph = new Graph(true);
          const m1 = new Mat(2, 2);
          m1.w[0] = 2;
          m1.w[1] = 3;
          m1.w[2] = 4;
          m1.w[3] = 5;
          const m2 = new Mat(2, 2);
          m2.w[0] = 2;
          m2.w[1] = 3;
          m2.w[2] = 4;
          m2.w[3] = 5;
          const out = graph.add(m1, m2);
          expect(graph.backprop.length).toBe(1);
          out.dw[0] = 2;
          out.dw[1] = 3;
          out.dw[2] = 4;
          out.dw[3] = 5;
          graph.backprop[0]();
          const expectedM1 = Mat.fromJSON(m1.toJSON());
          expectedM1.dw[0] = 2;
          expectedM1.dw[1] = 3;
          expectedM1.dw[2] = 4;
          expectedM1.dw[3] = 5;
          expect(m1).toEqual(expectedM1);
          const expectedM2 = Mat.fromJSON(m2.toJSON());
          expectedM2.dw[0] = 2;
          expectedM2.dw[1] = 3;
          expectedM2.dw[2] = 4;
          expectedM2.dw[3] = 5;
          expect(m2).toEqual(expectedM2);
        });
      });
    });
  });
  describe("dot", () => {
    it("returns added product of matrices", () => {
      const graph = new Graph();
      expect(graph.dot(Mat.fromJSON({
        n: 1,
        d: 1,
        w: [2],
      }), Mat.fromJSON({
        n: 1,
        d: 1,
        w: [2]
      }))).toEqual(Mat.fromJSON({
        n: 1,
        d: 1,
        w: [4],
      }));
    });
    describe("when used with needsBackprop = false", () => {
      it("does not push anything to this.backprop", () => {
        const graph = new Graph(false);
        graph.dot(new Mat(1, 1), new Mat(1, 1));
        expect(graph.backprop.length).toBe(0);
      });
    });
    describe("when used with needsBackprop = true", () => {
      describe("pushed function", () => {
        it("dot products out.dw into m1.dw and m2.dw", () => {
          const graph = new Graph(true);
          const m1 = new Mat(2, 2);
          m1.w[0] = 2;
          m1.w[1] = 3;
          m1.w[2] = 4;
          m1.w[3] = 5;
          const m2 = new Mat(2, 2);
          m2.w[0] = 2;
          m2.w[1] = 3;
          m2.w[2] = 4;
          m2.w[3] = 5;
          const out = graph.dot(m1, m2);
          expect(graph.backprop.length).toBe(1);
          out.dw[0] = 2;
          out.dw[1] = 3;
          out.dw[2] = 4;
          out.dw[3] = 5;
          graph.backprop[0]();
          const expectedM1 = Mat.fromJSON(m1.toJSON());
          expectedM1.dw[0] = 4;
          expectedM1.dw[1] = 6;
          expectedM1.dw[2] = 8;
          expectedM1.dw[3] = 10;
          expect(m1).toEqual(expectedM1);
          const expectedM2 = Mat.fromJSON(m2.toJSON());
          expectedM2.dw[0] = 4;
          expectedM2.dw[1] = 6;
          expectedM2.dw[2] = 8;
          expectedM2.dw[3] = 10;
          expect(m2).toEqual(expectedM2);
        });
      });
    });
  });
  describe("eltmul", () => {
    it("returns added product of matrices", () => {
      const graph = new Graph();
      expect(graph.eltmul(Mat.fromJSON({
        n: 1,
        d: 1,
        w: [2],
      }), Mat.fromJSON({
        n: 1,
        d: 1,
        w: [2]
      }))).toEqual(Mat.fromJSON({
        n: 1,
        d: 1,
        w: [4],
      }));
    });
    describe("when used with needsBackprop = false", () => {
      it("does not push anything to this.backprop", () => {
        const graph = new Graph(false);
        graph.eltmul(new Mat(1, 1), new Mat(1, 1));
        expect(graph.backprop.length).toBe(0);
      });
    });
    describe("when used with needsBackprop = true", () => {
      describe("pushed function", () => {
        it("element multiplies out.dw into m1.dw and m2.dw", () => {
          const graph = new Graph(true);
          const m1 = new Mat(2, 2);
          m1.w[0] = 2;
          m1.w[1] = 3;
          m1.w[2] = 4;
          m1.w[3] = 5;
          const m2 = new Mat(2, 2);
          m2.w[0] = 2;
          m2.w[1] = 3;
          m2.w[2] = 4;
          m2.w[3] = 5;
          const out = graph.eltmul(m1, m2);
          expect(graph.backprop.length).toBe(1);
          out.dw[0] = 2;
          out.dw[1] = 3;
          out.dw[2] = 4;
          out.dw[3] = 5;
          graph.backprop[0]();
          const expectedM1 = Mat.fromJSON(m1.toJSON());
          expectedM1.dw[0] = 4;
          expectedM1.dw[1] = 9;
          expectedM1.dw[2] = 16;
          expectedM1.dw[3] = 25;
          expect(m1).toEqual(expectedM1);
          const expectedM2 = Mat.fromJSON(m2.toJSON());
          expectedM2.dw[0] = 4;
          expectedM2.dw[1] = 9;
          expectedM2.dw[2] = 16;
          expectedM2.dw[3] = 25;
          expect(m2).toEqual(expectedM2);
        });
      });
    });
  });
  describe("softmax", () => {
    it("returns added product of matrices", () => {
      const graph = new Graph();
      expect(graph.softmax(Mat.fromJSON({
        n: 1,
        d: 1,
        w: [2],
      }))).toEqual(Mat.fromJSON({
        n: 1,
        d: 1,
        w: [1],
      }));
    });
  });
});
