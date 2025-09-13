import { useState, useEffect } from "react";
import { useApi } from "../context/ApiContext";

export const useModelData = () => {
  const { checkMLHealth, fetchCategories } = useApi();

  const [mlHealth, setMlHealth] = useState(null);
  const [categories, setCategories] = useState([]);

  useEffect(() => {
    const loadInitialData = async () => {
      try {
        const [healthData, categoriesData] = await Promise.all([
          checkMLHealth(),
          fetchCategories(),
        ]);
        setMlHealth(healthData);
        // Transform categories from object to array format
        const categoriesObj = categoriesData.categories || {};
        const categoriesArray = Object.entries(categoriesObj).map(([id, name]) => ({
          id: parseInt(id),
          name: name
        }));
        setCategories(categoriesArray);
      } catch (err) {
        // Error handled by ApiContext
      }
    };

    loadInitialData();
  }, [checkMLHealth, fetchCategories]);

  return {
    mlHealth,
    categories,
  };
};