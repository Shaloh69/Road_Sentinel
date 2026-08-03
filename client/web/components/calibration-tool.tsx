"use client";

import { useRef, useState } from "react";
import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
} from "@heroui/modal";
import { Button } from "@heroui/button";
import { Input } from "@heroui/input";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3001";

const POINT_LABELS = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"];

interface Point {
  x: number;
  y: number;
}

interface HomographyPoints {
  image_points: [number, number][];
  real_points: [number, number][];
}

interface CalibrationToolProps {
  isOpen: boolean;
  onClose: () => void;
  cameraId: string;
  cameraName: string;
  existing: HomographyPoints | null;
  onSaved: () => void;
}

// Mark 4 corners of a known rectangle on the road (e.g. a lane marking or
// measured section) in the order Top-left, Top-right, Bottom-right,
// Bottom-left — the same convention inference/camera_calibration.py uses.
export function CalibrationTool({
  isOpen,
  onClose,
  cameraId,
  cameraName,
  existing,
  onSaved,
}: CalibrationToolProps) {
  const [points, setPoints] = useState<Point[]>([]);
  const [width, setWidth] = useState("3.5");
  const [length, setLength] = useState("10");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const streamUrl = `${API}/api/cameras/${cameraId}/stream`;

  const handleImageClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (points.length >= 4 || !imgRef.current) return;
    const rect = imgRef.current.getBoundingClientRect();
    const displayX = e.clientX - rect.left;
    const displayY = e.clientY - rect.top;

    // Scale from displayed size to the video's native resolution so stored
    // points stay correct regardless of how big the <img> renders in the browser.
    const scaleX = imgRef.current.naturalWidth / rect.width;
    const scaleY = imgRef.current.naturalHeight / rect.height;

    setPoints((prev) => [
      ...prev,
      { x: Math.round(displayX * scaleX), y: Math.round(displayY * scaleY) },
    ]);
  };

  const reset = () => {
    setPoints([]);
    setMessage(null);
  };

  const save = async () => {
    const w = parseFloat(width);
    const l = parseFloat(length);

    if (points.length !== 4 || !w || !l || w <= 0 || l <= 0) {
      setMessage("Click all 4 points and enter a valid width/length first.");

      return;
    }
    setSaving(true);
    setMessage(null);
    const homography_points: HomographyPoints = {
      image_points: points.map((p) => [p.x, p.y]),
      real_points: [
        [0, 0],
        [w, 0],
        [w, l],
        [0, l],
      ],
    };

    try {
      const res = await fetch(`${API}/api/cameras/${cameraId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ homography_points }),
      });
      const json = await res.json();

      if (json.success) {
        setMessage(
          "Calibration saved — speed estimates will use it immediately.",
        );
        onSaved();
      } else {
        setMessage(json.error ?? "Save failed");
      }
    } catch {
      setMessage("Network error — check server connection");
    } finally {
      setSaving(false);
    }
  };

  const clearCalibration = async () => {
    setSaving(true);
    try {
      const res = await fetch(`${API}/api/cameras/${cameraId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ homography_points: null }),
      });
      const json = await res.json();

      if (json.success) {
        setMessage("Calibration cleared — falling back to Pixels Per Meter.");
        reset();
        onSaved();
      }
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal isOpen={isOpen} scrollBehavior="inside" size="3xl" onClose={onClose}>
      <ModalContent>
        <ModalHeader className="flex flex-col gap-1">
          Calibration Tool — {cameraName}
        </ModalHeader>
        <ModalBody>
          <p className="text-sm text-fg-muted">
            Click the 4 corners of a known rectangle on the road, in order:{" "}
            <strong>{POINT_LABELS.join(" → ")}</strong>. A lane marking or a
            measured section works well. Enter the rectangle&apos;s real width
            and length in meters, then save.
          </p>
          {existing && points.length === 0 && (
            <p className="text-xs text-success">
              This camera is already calibrated. Clicking 4 new points will
              replace the existing calibration.
            </p>
          )}
          {/* Click-to-place-point on a live image has no meaningful keyboard
              equivalent (there's no discrete tab order over pixel coordinates) —
              same accepted tradeoff as the admin terminal's click-to-focus div. */}
          {/* eslint-disable-next-line jsx-a11y/no-static-element-interactions, jsx-a11y/click-events-have-key-events */}
          <div
            className="relative w-full aspect-video bg-black rounded-lg overflow-hidden border border-border"
            onClick={handleImageClick}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              ref={imgRef}
              alt={cameraName}
              className="w-full h-full object-contain cursor-crosshair"
              src={streamUrl}
            />
            {points.map((p, i) => (
              <div
                key={i}
                className="absolute w-3 h-3 -ml-1.5 -mt-1.5 rounded-full bg-brand border-2 border-white pointer-events-none"
                style={{
                  left: `${(p.x / (imgRef.current?.naturalWidth || 1)) * 100}%`,
                  top: `${(p.y / (imgRef.current?.naturalHeight || 1)) * 100}%`,
                }}
              >
                <span className="absolute left-4 top-0 text-[10px] font-bold text-white bg-black/70 px-1 rounded whitespace-nowrap">
                  {i + 1}. {POINT_LABELS[i]}
                </span>
              </div>
            ))}
          </div>
          <div className="grid grid-cols-2 gap-4">
            <Input
              classNames={{
                label: "text-fg-muted",
                input: "text-fg",
                inputWrapper: "bg-surface-2 border-border",
              }}
              label="Rectangle width (m)"
              type="number"
              value={width}
              onChange={(e) => setWidth(e.target.value)}
            />
            <Input
              classNames={{
                label: "text-fg-muted",
                input: "text-fg",
                inputWrapper: "bg-surface-2 border-border",
              }}
              label="Rectangle length (m)"
              type="number"
              value={length}
              onChange={(e) => setLength(e.target.value)}
            />
          </div>
          <p className="text-xs text-fg-muted">
            {points.length}/4 points marked.{" "}
            {points.length > 0 && (
              <button className="underline hover:text-fg" onClick={reset}>
                Start over
              </button>
            )}
          </p>
          {message && <p className="text-sm text-brand">{message}</p>}
        </ModalBody>
        <ModalFooter>
          {existing && (
            <Button
              className="bg-danger/20 text-danger border border-danger/30"
              isDisabled={saving}
              onClick={clearCalibration}
            >
              Clear Calibration
            </Button>
          )}
          <Button variant="light" onClick={onClose}>
            Close
          </Button>
          <Button
            className="bg-brand text-brand-foreground font-semibold"
            isDisabled={points.length !== 4}
            isLoading={saving}
            onClick={save}
          >
            Save Calibration
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}

export function CalibrationGuide({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  return (
    <Modal isOpen={isOpen} scrollBehavior="inside" size="2xl" onClose={onClose}>
      <ModalContent>
        <ModalHeader>Camera Calibration Guide</ModalHeader>
        <ModalBody className="text-sm text-fg-muted space-y-4 pb-6">
          <div>
            <p className="font-semibold text-fg mb-1">Why calibrate at all?</p>
            <p>
              Without calibration, speed is estimated from raw pixel movement
              using a single flat &quot;pixels per meter&quot; value. Because of
              camera perspective, a vehicle far from the camera covers fewer
              pixels per real meter than one close to the camera — so that
              estimate is systematically wrong depending on where in frame the
              vehicle is. Calibrating with 4 known points corrects for this.
            </p>
          </div>
          <div>
            <p className="font-semibold text-fg mb-1">
              Step 1 — Find a known rectangle on the road
            </p>
            <ul className="list-disc list-inside space-y-1">
              <li>Lane marking (commonly ~3.5m wide in the Philippines)</li>
              <li>Parking space (~2.5m × 5m typical)</li>
              <li>Pedestrian crossing stripes (~2m × 4m)</li>
            </ul>
          </div>
          <div>
            <p className="font-semibold text-fg mb-1">
              Step 2 — Measure its real dimensions
            </p>
            <p>
              Use a measuring tape or laser distance meter. Precise,
              well-distributed points matter more than which algorithm is used —
              spend the effort here.
            </p>
          </div>
          <div>
            <p className="font-semibold text-fg mb-1">
              Step 3 — Mark the 4 corners in the Calibration Tool
            </p>
            <p>
              Click in order: top-left, top-right, bottom-right, bottom-left.
              Order matters — it must match the width/length you enter.
            </p>
          </div>
          <div>
            <p className="font-semibold text-fg mb-1">Step 4 — Save</p>
            <p>
              The tool solves a perspective transform from your 4 points and
              stores it on the camera. Speed estimates switch to it immediately
              — no restart needed.
            </p>
          </div>
          <div className="p-3 bg-surface-2/60 rounded-lg border border-border">
            <p className="text-fg-muted text-xs">
              💡 For the Busay blind curve specifically: a 3.5m × 10m section
              (about one lane width × two car lengths) marked with road paint or
              temporary markers works well.
            </p>
          </div>
        </ModalBody>
      </ModalContent>
    </Modal>
  );
}
