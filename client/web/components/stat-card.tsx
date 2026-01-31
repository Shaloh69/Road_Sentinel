import { Card, CardBody } from "@heroui/card";
import { ReactNode } from "react";

interface StatCardProps {
  title: string;
  value: string | number;
  icon?: ReactNode;
  trend?: {
    value: string;
    isPositive: boolean;
  };
  subtitle?: string;
}

export const StatCard = ({ title, value, icon, trend, subtitle }: StatCardProps) => {
  return (
    <Card className="bg-white/10 backdrop-blur-md border border-white/20 shadow-xl hover:bg-white/15 transition-all">
      <CardBody className="p-6">
        <div className="flex justify-between items-start">
          <div className="flex-1">
            <p className="text-sm text-white/80 font-medium uppercase tracking-wide">
              {title}
            </p>
            <p className="text-4xl font-bold text-white mt-2">{value}</p>
            {subtitle && (
              <p className="text-xs text-white/70 mt-1">{subtitle}</p>
            )}
            {trend && (
              <div className="flex items-center gap-1 mt-3">
                <span
                  className={`text-sm font-semibold ${
                    trend.isPositive ? "text-green-300" : "text-red-300"
                  }`}
                >
                  {trend.isPositive ? "↑" : "↓"} {trend.value}
                </span>
                <span className="text-xs text-white/60">vs yesterday</span>
              </div>
            )}
          </div>
          {icon && (
            <div className="bg-[#ED9E59]/30 p-3 rounded-xl backdrop-blur-sm">
              <div className="text-[#ED9E59] w-6 h-6">{icon}</div>
            </div>
          )}
        </div>
      </CardBody>
    </Card>
  );
};
