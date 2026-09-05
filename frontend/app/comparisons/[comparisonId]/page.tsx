import { ComparisonView } from "@/components/comparison/ComparisonView";

export default async function ComparisonPage({
  params,
}: {
  params: Promise<{ comparisonId: string }>;
}) {
  const { comparisonId } = await params;
  return <ComparisonView comparisonId={comparisonId} />;
}
