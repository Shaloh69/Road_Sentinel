"use client";

import { useEffect, useState } from "react";
import { Card, CardBody, CardHeader } from "@heroui/card";
import { Button } from "@heroui/button";
import { Switch } from "@heroui/switch";
import { addToast } from "@heroui/toast";

import {
  DEFAULT_SETTINGS,
  getSettings,
  saveSettings,
  playAlertSound,
  Settings,
} from "@/lib/settings";

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setSettings(getSettings());
  }, []);

  const handleSave = () => {
    saveSettings(settings);
    setSaved(true);
    addToast({
      title: "Settings saved",
      description: "Preferences updated for this browser",
      color: "success",
      timeout: 3000,
    });
    setTimeout(() => setSaved(false), 2000);
  };

  const handleReset = () => {
    setSettings(DEFAULT_SETTINGS);
    saveSettings(DEFAULT_SETTINGS);
    addToast({
      title: "Settings reset",
      description: "Restored to defaults",
      color: "primary",
      timeout: 3000,
    });
  };

  return (
    <div className="min-h-screen p-6">
      <div className="mb-6">
        <h1 className="text-4xl font-heading font-bold text-fg mb-2">
          Settings
        </h1>
        <p className="text-fg-muted">
          Configure system preferences and detection parameters
        </p>
      </div>

      <div className="space-y-6">
        {/* Alert Settings */}
        <Card className="bg-surface/80 backdrop-blur-md border border-border shadow-xl">
          <CardHeader className="bg-surface-2/60 backdrop-blur-sm px-4 py-3 border-b border-border">
            <h3 className="text-xl font-heading font-bold text-fg">
              Alert & Notification Settings
            </h3>
          </CardHeader>
          <CardBody className="p-6">
            <div className="space-y-4">
              <div className="flex justify-between items-center p-4 bg-surface-2/60 backdrop-blur-sm rounded-lg border border-border">
                <div>
                  <p className="text-fg font-medium">Email Notifications</p>
                  <p className="text-fg-muted text-sm">
                    Preference saved for when email alerts ship (Phase 2 — no
                    email sending exists yet)
                  </p>
                </div>
                <Switch
                  color="primary"
                  isSelected={settings.emailNotifications}
                  onValueChange={(v) =>
                    setSettings((s) => ({ ...s, emailNotifications: v }))
                  }
                />
              </div>
              <div className="flex justify-between items-center p-4 bg-surface-2/60 backdrop-blur-sm rounded-lg border border-border">
                <div>
                  <p className="text-fg font-medium">Sound Alerts</p>
                  <p className="text-fg-muted text-sm">
                    Play a sound when a critical incident arrives on the
                    Incidents page
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {settings.soundAlerts && (
                    <button
                      className="text-xs text-fg-muted hover:text-fg underline"
                      onClick={playAlertSound}
                    >
                      Test
                    </button>
                  )}
                  <Switch
                    color="primary"
                    isSelected={settings.soundAlerts}
                    onValueChange={(v) =>
                      setSettings((s) => ({ ...s, soundAlerts: v }))
                    }
                  />
                </div>
              </div>
            </div>
          </CardBody>
        </Card>

        {/* Save Button */}
        <div className="flex justify-end items-center gap-3">
          {saved && (
            <span className="text-sm text-success">Settings saved</span>
          )}
          <Button
            className="bg-surface-2/80 backdrop-blur-md text-fg font-semibold hover:bg-surface-2 border border-border shadow-lg"
            onClick={handleReset}
          >
            Reset to Defaults
          </Button>
          <Button
            className="font-semibold shadow-lg"
            color="primary"
            onClick={handleSave}
          >
            Save All Settings
          </Button>
        </div>
      </div>
    </div>
  );
}
