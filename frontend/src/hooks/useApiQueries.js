import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useApi } from '../context/ApiContext';

// Query keys for consistent caching
export const queryKeys = {
    statistics: ['statistics'],
    countries: ['countries'],
    categories: ['categories'],
    trendingVideos: (filters) => ['trendingVideos', filters],
    databaseStats: ['databaseStats'],
    mlHealth: ['mlHealth'],
    prediction: (url) => ['prediction', url],
};

// Custom hooks for data fetching
export const useStatistics = (country = null) => {
    const { fetchStatistics } = useApi();

    return useQuery({
        queryKey: [...queryKeys.statistics, country],
        queryFn: () => fetchStatistics(country),
        staleTime: 2 * 60 * 1000, // 2 minutes - statistics change frequently
        gcTime: 5 * 60 * 1000,
    });
};

export const useCountries = () => {
    const { fetchCountries } = useApi();

    return useQuery({
        queryKey: queryKeys.countries,
        queryFn: fetchCountries,
        select: (data) => data.countries || [],
        staleTime: 30 * 60 * 1000, // 30 minutes - countries rarely change
        gcTime: 60 * 60 * 1000, // 1 hour
    });
};

export const useCategories = () => {
    const { fetchCategories } = useApi();

    return useQuery({
        queryKey: queryKeys.categories,
        queryFn: fetchCategories,
        select: (data) => data.categories || {},
        staleTime: 30 * 60 * 1000, // 30 minutes - categories rarely change
        gcTime: 60 * 60 * 1000,
    });
};

export const useTrendingVideos = (filters = {}) => {
    const { fetchTrendingVideos } = useApi();

    return useQuery({
        queryKey: queryKeys.trendingVideos(filters),
        queryFn: () => fetchTrendingVideos(filters),
        staleTime: 1 * 60 * 1000, // 1 minute - trending data changes frequently
        gcTime: 5 * 60 * 1000,
    });
};

export const useDatabaseStats = () => {
    const { fetchDatabaseStats } = useApi();

    return useQuery({
        queryKey: queryKeys.databaseStats,
        queryFn: fetchDatabaseStats,
        staleTime: 5 * 60 * 1000, // 5 minutes - DB stats don't change often
        gcTime: 15 * 60 * 1000,
    });
};

export const useMlHealth = () => {
    const { checkMLHealth } = useApi();

    return useQuery({
        queryKey: queryKeys.mlHealth,
        queryFn: checkMLHealth,
        staleTime: 10 * 60 * 1000, // 10 minutes - ML models don't change often
        gcTime: 30 * 60 * 1000,
        refetchInterval: 5 * 60 * 1000, // Refetch every 5 minutes
    });
};

// Prediction mutation (for POST requests)
export const usePredictionMutation = () => {
    const { predictByUrl } = useApi();
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: predictByUrl,
        onSuccess: (data, url) => {
            // Cache the prediction result
            queryClient.setQueryData(queryKeys.prediction(url), data);
        },
    });
};

// Hook to get cached prediction
export const usePrediction = (url) => {
    const { predictByUrl } = useApi();

    return useQuery({
        queryKey: queryKeys.prediction(url),
        queryFn: () => {
            if (!url || !url.trim()) {
                return Promise.resolve(null);
            }
            return predictByUrl(url);
        },
        enabled: !!(url && url.trim()), // Only run if URL exists and is not just whitespace
        staleTime: 30 * 60 * 1000, // 30 minutes - predictions can be cached
        gcTime: 60 * 60 * 1000,
    });
};