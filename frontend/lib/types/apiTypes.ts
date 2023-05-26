export type Group = {
  name: string;
};

export type Frame = {
  id: string;
  number: number;
  group: Group;
  people: Array<string>;
};

export type Person = {
  id: string;
  keypoints: Keypoint;
  box: BoundingBox;
};

export type Point = {
  x: number;
  y: number;
};

export type ProbabilisticPoint = Point & {
  p: number;
};

export type Keypoint = {
  nose: ProbabilisticPoint;
  neck: ProbabilisticPoint;
  r_shoulder: ProbabilisticPoint;
  r_elbow: ProbabilisticPoint;
  r_wrist: ProbabilisticPoint;
  l_shoulder: ProbabilisticPoint;
  l_elbow: ProbabilisticPoint;
  l_wrist: ProbabilisticPoint;
  midhip: ProbabilisticPoint;
  r_hip: ProbabilisticPoint;
  r_knee: ProbabilisticPoint;
  r_ankle: ProbabilisticPoint;
  l_hip: ProbabilisticPoint;
  l_knee: ProbabilisticPoint;
  l_ankle: ProbabilisticPoint;
  r_eye: ProbabilisticPoint;
  l_eye: ProbabilisticPoint;
  r_ear: ProbabilisticPoint;
  l_ear: ProbabilisticPoint;
  l_bigtoe: ProbabilisticPoint;
  l_smalltoe: ProbabilisticPoint;
  l_heel: ProbabilisticPoint;
  r_bigtoe: ProbabilisticPoint;
  r_smalltoe: ProbabilisticPoint;
  r_hell: ProbabilisticPoint;
};

export type BoundingBox = {
  id: number;
  min: Point;
  max: Point;
};
