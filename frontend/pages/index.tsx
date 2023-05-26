import { useState } from "react";
import Image from "next/image";
import { Group } from "@/lib/types/apiTypes";
import axios from "axios";
import { GetStaticProps } from "next";

export const getStaticProps: GetStaticProps = async (ctx) => {
  const res = await axios.get<Group[]>("http://backend:8000/api/groups/");

  return {
    props: {
      groups: res.data,
    },
  };
};

type PageProp = {
  groups: Group[];
};

export default function Home({ groups }: PageProp) {
  return (
    <>
      {groups.map((group) => (
        <p key={`group-${group.name}`}>{group.name}</p>
      ))}
    </>
  );
}
