import { useState } from "react";
import Image from "next/image";

export default function Home() {
  const [frameNumber, setFrameNumber] = useState<number>(0);

  return (
    <>
      <img
        src={`/outputs/video/ID/frame_000000000${frameNumber}.jpg`}
        alt="test"
        width={1080}
        height={720}
      />
      <button
        className="bg-blue-200 rounded-full p-3 transition hover:bg-blue-400 border-2 border-black"
        onClick={() => {
          setFrameNumber((prev) => prev + 1);
        }}
      >
        count up
      </button>
      Frame: {frameNumber}
    </>
  );
}
