"use client";

import { useEffect } from "react";
import { Button } from "@heroui/button";

export default function Error({
  error,
  reset,
}: {
  error: Error;
  reset: () => void;
}) {
  useEffect(() => {
    // Log the error to an error reporting service
    /* eslint-disable no-console */
    console.error(error);
  }, [error]);

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="w-full max-w-md text-center p-8 bg-surface/80 backdrop-blur-md border border-border rounded-2xl shadow-xl">
        <p className="text-danger text-sm font-mono uppercase tracking-widest mb-2">
          Error
        </p>
        <h2 className="text-2xl font-heading font-bold text-fg mb-2">
          Something went wrong
        </h2>
        <p className="text-fg-muted text-sm mb-6">
          {error.message || "An unexpected error occurred."}
        </p>
        <Button
          className="font-semibold"
          color="primary"
          onClick={() => reset()}
        >
          Try again
        </Button>
      </div>
    </div>
  );
}
