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
    <Card className="bg-gradient-to-br from-[#44174E] to-[#862249] border-none shadow-lg">
      <CardBody className="p-6">
        <div className="flex justify-between items-start">
          <div className="flex-1">
            <p className="text-sm text-[#E8BCB8] font-medium uppercase tracking-wide">
              {title}
            </p>
            <p className="text-4xl font-bold text-[#ED9E59] mt-2">{value}</p>
            {subtitle && (
              <p className="text-xs text-[#E8BCB8] mt-1 opacity-80">{subtitle}</p>
            )}
            {trend && (
              <div className="flex items-center gap-1 mt-3">
                <span
                  className={`text-sm font-semibold ${
                    trend.isPositive ? "text-green-400" : "text-red-400"
                  }`}
                >
                  {trend.isPositive ? "↑" : "↓"} {trend.value}
                </span>
                <span className="text-xs text-[#E8BCB8] opacity-70">vs yesterday</span>
              </div>
            )}
          </div>
          {icon && (
            <div className="bg-[#ED9E59] bg-opacity-20 p-3 rounded-lg">
              <div className="text-[#ED9E59] w-6 h-6">{icon}</div>
            </div>
          )}
        </div>
      </CardBody>
    </Card>
  );
};
