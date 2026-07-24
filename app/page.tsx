import type { Metadata } from "next";
import { StudyHub } from "./StudyHub";

export const metadata: Metadata = {
  title: "LLM RL 学习路径｜4 周加速版",
  description: "面向算法工程师的强化学习大模型后训练学习站。",
};

export default function Home() {
  return <StudyHub />;
}
