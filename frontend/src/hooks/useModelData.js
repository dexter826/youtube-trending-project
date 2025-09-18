import { useMlHealth, useCategories } from "./useApiQueries";

export const useModelData = () => {
  const { data: mlHealth, isLoading: mlHealthLoading, error: mlHealthError } = useMlHealth();
  const { data: categoriesData, isLoading: categoriesLoading, error: categoriesError } = useCategories();

  // Transform categories from object to array format
  const categories = categoriesData?.categories
    ? Object.entries(categoriesData.categories).map(([id, name]) => ({
      id: parseInt(id),
      name: name
    }))
    : [];

  return {
    mlHealth,
    categories,
    loading: mlHealthLoading || categoriesLoading,
    error: mlHealthError || categoriesError,
  };
};