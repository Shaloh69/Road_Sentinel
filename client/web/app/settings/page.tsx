import { Card, CardBody, CardHeader } from "@heroui/card";
import { Button } from "@heroui/button";
import { Switch } from "@heroui/switch";

export default function SettingsPage() {
  return (
    <div className="min-h-screen p-6">
      <div className="mb-6">
        <h1 className="text-4xl font-bold text-[#ED9E59] mb-2">Settings</h1>
        <p className="text-[#E8BCB8]">Configure system preferences and detection parameters</p>
      </div>

      <div className="space-y-6">
        {/* Alert Settings */}
        <Card className="bg-[#1B1931] border-2 border-[#44174E]">
          <CardHeader className="bg-gradient-to-r from-[#44174E] to-[#862249] px-4 py-3">
            <h3 className="text-xl font-bold text-[#ED9E59]">Alert & Notification Settings</h3>
          </CardHeader>
          <CardBody className="p-6">
            <div className="space-y-4">
              <div className="flex justify-between items-center p-4 bg-[#44174E] rounded-lg">
                <div>
                  <p className="text-[#E8BCB8] font-medium">Email Notifications</p>
                  <p className="text-[#E8BCB8] text-sm opacity-70">Receive alerts via email</p>
                </div>
                <Switch
                  defaultSelected
                  classNames={{
                    wrapper: "group-data-[selected=true]:bg-[#ED9E59]",
                  }}
                />
              </div>
              <div className="flex justify-between items-center p-4 bg-[#44174E] rounded-lg">
                <div>
                  <p className="text-[#E8BCB8] font-medium">Sound Alerts</p>
                  <p className="text-[#E8BCB8] text-sm opacity-70">Play sound for critical incidents</p>
                </div>
                <Switch
                  defaultSelected
                  classNames={{
                    wrapper: "group-data-[selected=true]:bg-[#ED9E59]",
                  }}
                />
              </div>
            </div>
          </CardBody>
        </Card>

        {/* Save Button */}
        <div className="flex justify-end gap-3">
          <Button className="bg-[#862249] text-[#E8BCB8] font-semibold hover:bg-[#A34054]">
            Reset to Defaults
          </Button>
          <Button className="bg-[#ED9E59] text-[#1B1931] font-semibold hover:bg-[#A34054]">
            Save All Settings
          </Button>
        </div>
      </div>
    </div>
  );
}
