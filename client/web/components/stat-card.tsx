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

export const StatCard = ({
  title,
  value,
  icon,
  trend,
  subtitle,
}: StatCardProps) => {
  return (
    <Card className="bg-surface/80 backdrop-blur-md border border-border shadow-xl hover:bg-surface-2/80 transition-colors duration-150 ease-standard">
      <CardBody className="p-6">
        <div className="flex justify-between items-start">
          <div className="flex-1">
            <p className="text-sm text-fg-muted font-medium uppercase tracking-wide">
              {title}
            </p>
            <p className="text-4xl font-heading font-bold text-fg mt-2">
              {value}
            </p>
            {subtitle && (
              <p className="text-xs text-fg-muted mt-1">{subtitle}</p>
            )}
            {trend && (
              <div className="flex items-center gap-1 mt-3">
                <span
                  className={`text-sm font-semibold ${
                    trend.isPositive ? "text-success" : "text-danger"
                  }`}
                >
                  {trend.isPositive ? "↑" : "↓"} {trend.value}
                </span>
                <span className="text-xs text-fg-muted">vs yesterday</span>
              </div>
            )}
          </div>
          {icon && (
            <div className="bg-brand/15 p-3 rounded-xl backdrop-blur-sm">
              <div className="text-brand w-6 h-6">{icon}</div>
            </div>
          )}
        </div>
      </CardBody>
    </Card>
  );
};
