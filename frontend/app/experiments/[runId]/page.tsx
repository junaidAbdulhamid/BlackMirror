import { ExperimentView } from "@/components/ExperimentView";

export default async function ExperimentPage({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = await params;
  return <ExperimentView runId={runId} />;
}
