import React, { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { BarChart3, TrendingUp, Brain, Menu, X, Youtube } from "lucide-react";
import { useApi } from "../context/ApiContext";

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();
  const { apiHealth } = useApi();

  const navigation = [
    { name: "Dashboard", href: "/", icon: BarChart3 },
    { name: "Phân tích Trending", href: "/analysis", icon: TrendingUp },
    { name: "Dự đoán Video", href: "/prediction", icon: Brain },
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <nav className="sticky top-0 z-50 bg-red-600 shadow-lg border-b border-red-700">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Logo and brand */}
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <div className="flex items-center justify-center w-10 h-10 bg-white rounded-lg">
                <Youtube className="w-6 h-6 text-red-600" />
              </div>
              <div className="hidden sm:block">
                <h1 className="text-xl font-bold text-white">
                  YouTube Analytics
                </h1>
                <p className="text-xs text-red-200">Big Data & ML Platform</p>
              </div>
            </Link>
          </div>

          {/* Desktop navigation */}
          <div className="hidden md:flex items-center space-x-1">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200 ${
                    isActive(item.href)
                      ? "bg-white text-red-600 border border-red-300"
                      : "text-white hover:text-red-100 hover:bg-red-700"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </div>

          {/* API Status and Mobile menu button */}
          <div className="flex items-center space-x-4">
            {/* API Status Indicator */}
            <div className="hidden sm:flex items-center space-x-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  apiHealth?.status === "healthy"
                    ? "bg-green-400"
                    : "bg-red-400"
                }`}
              />
              <span className="text-xs text-red-200">
                API {apiHealth?.status === "healthy" ? "Online" : "Offline"}
              </span>
            </div>

            {/* Mobile menu button */}
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="md:hidden inline-flex items-center justify-center p-2 rounded-md text-white hover:text-red-100 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
            >
              {isOpen ? (
                <X className="block h-6 w-6" />
              ) : (
                <Menu className="block h-6 w-6" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {isOpen && (
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-red-700 border-t border-red-500">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={() => setIsOpen(false)}
                  className={`flex items-center space-x-3 px-3 py-2 rounded-md text-base font-medium transition-colors duration-200 ${
                    isActive(item.href)
                      ? "bg-white text-red-600"
                      : "text-white hover:text-red-100 hover:bg-red-800"
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{item.name}</span>
                </Link>
              );
            })}

            {/* Mobile API Status */}
            <div className="flex items-center space-x-3 px-3 py-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  apiHealth?.status === "healthy"
                    ? "bg-green-400"
                    : "bg-red-400"
                }`}
              />
              <span className="text-sm text-red-200">
                API Status:{" "}
                {apiHealth?.status === "healthy" ? "Online" : "Offline"}
              </span>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
