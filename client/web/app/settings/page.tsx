import { Card, CardBody, CardHeader } from "@heroui/card";
import { Button } from "@heroui/button";
import { Switch } from "@heroui/switch";

export default function SettingsPage() {
  return (
    <div className="min-h-screen p-6">
      <div className="mb-6">
        <h1 className="text-4xl font-bold text-white mb-2">Settings</h1>
        <p className="text-white/70">Configure system preferences and detection parameters</p>
      </div>

      <div className="space-y-6">
        {/* Alert Settings */}
        <Card className="bg-white/10 backdrop-blur-md border border-white/20 shadow-xl">
          <CardHeader className="bg-white/10 backdrop-blur-sm px-4 py-3 border-b border-white/10">
            <h3 className="text-xl font-bold text-white">Alert & Notification Settings</h3>
          </CardHeader>
          <CardBody className="p-6">
            <div className="space-y-4">
              <div className="flex justify-between items-center p-4 bg-white/10 backdrop-blur-sm rounded-lg border border-white/10">
                <div>
                  <p className="text-white/90 font-medium">Email Notifications</p>
                  <p className="text-white/70 text-sm">Receive alerts via email</p>
                </div>
                <Switch
                  defaultSelected
                  classNames={{
                    wrapper: "group-data-[selected=true]:bg-white/30",
                  }}
                />
              </div>
              <div className="flex justify-between items-center p-4 bg-white/10 backdrop-blur-sm rounded-lg border border-white/10">
                <div>
                  <p className="text-white/90 font-medium">Sound Alerts</p>
                  <p className="text-white/70 text-sm">Play sound for critical incidents</p>
                </div>
                <Switch
                  defaultSelected
                  classNames={{
                    wrapper: "group-data-[selected=true]:bg-white/30",
                  }}
                />
              </div>
            </div>
          </CardBody>
        </Card>

        {/* Save Button */}
        <div className="flex justify-end gap-3">
          <Button className="bg-white/20 backdrop-blur-md text-white font-semibold hover:bg-white/30 border border-white/20 shadow-lg">
            Reset to Defaults
          </Button>
          <Button className="bg-white text-[#1B1931] font-semibold hover:bg-white/90 shadow-lg">
            Save All Settings
          </Button>
        </div>
      </div>
    </div>
  );
}
