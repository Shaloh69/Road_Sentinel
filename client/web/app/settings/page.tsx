import { Card, CardBody, CardHeader } from "@heroui/card";
import { Button } from "@heroui/button";
import { Input } from "@heroui/input";
import { Switch } from "@heroui/switch";
import { Select, SelectItem } from "@heroui/select";
import { Slider } from "@heroui/slider";

export default function SettingsPage() {
  return (
    <div className="min-h-screen p-6">
      <div className="mb-6">
        <h1 className="text-4xl font-bold text-[#ED9E59] mb-2">Settings</h1>
        <p className="text-[#E8BCB8]">Configure system preferences and detection parameters</p>
      </div>

      <div className="space-y-6">
        {/* Model Settings */}
        <Card className="bg-[#1B1931] border-2 border-[#44174E]">
          <CardHeader className="bg-gradient-to-r from-[#44174E] to-[#862249] px-4 py-3">
            <h3 className="text-xl font-bold text-[#ED9E59]">AI Model Settings</h3>
          </CardHeader>
          <CardBody className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h4 className="text-lg font-semibold text-[#ED9E59]">Vehicle Detection Model</h4>
                <Select
                  label="Model Version"
                  defaultSelectedKeys={["v1"]}
                  classNames={{
                    label: "text-[#E8BCB8]",
                    trigger: "bg-[#44174E] border-[#862249] text-[#E8BCB8]",
                  }}
                >
                  <SelectItem key="v1">v1 - Production</SelectItem>
                  <SelectItem key="v2">v2 - Testing</SelectItem>
                </Select>
                <Slider
                  label="Confidence Threshold"
                  step={0.01}
                  minValue={0}
                  maxValue={1}
                  defaultValue={0.75}
                  className="max-w-md"
                  classNames={{
                    label: "text-[#E8BCB8]",
                    value: "text-[#ED9E59]",
                    track: "bg-[#44174E]",
                    filler: "bg-[#ED9E59]",
                  }}
                />
              </div>

              <div className="space-y-4">
                <h4 className="text-lg font-semibold text-[#ED9E59]">Crash Detection Model</h4>
                <Select
                  label="Model Version"
                  defaultSelectedKeys={["v1"]}
                  classNames={{
                    label: "text-[#E8BCB8]",
                    trigger: "bg-[#44174E] border-[#862249] text-[#E8BCB8]",
                  }}
                >
                  <SelectItem key="v1">v1 - Production</SelectItem>
                  <SelectItem key="v2">v2 - Testing</SelectItem>
                </Select>
                <Slider
                  label="Sensitivity"
                  step={0.01}
                  minValue={0}
                  maxValue={1}
                  defaultValue={0.85}
                  className="max-w-md"
                  classNames={{
                    label: "text-[#E8BCB8]",
                    value: "text-[#ED9E59]",
                    track: "bg-[#44174E]",
                    filler: "bg-[#ED9E59]",
                  }}
                />
              </div>
            </div>
          </CardBody>
        </Card>

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
                  <p className="text-[#E8BCB8] font-medium">SMS Notifications</p>
                  <p className="text-[#E8BCB8] text-sm opacity-70">Receive critical alerts via SMS</p>
                </div>
                <Switch
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

              <Input
                label="Alert Email"
                type="email"
                defaultValue="admin@roadsentinel.ph"
                classNames={{
                  label: "text-[#E8BCB8]",
                  input: "text-[#E8BCB8]",
                  inputWrapper: "bg-[#44174E] border-[#862249]",
                }}
              />
            </div>
          </CardBody>
        </Card>

        {/* System Settings */}
        <Card className="bg-[#1B1931] border-2 border-[#44174E]">
          <CardHeader className="bg-gradient-to-r from-[#44174E] to-[#862249] px-4 py-3">
            <h3 className="text-xl font-bold text-[#ED9E59]">System Settings</h3>
          </CardHeader>
          <CardBody className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <Input
                  label="System Name"
                  defaultValue="Road Sentinel - Busay"
                  classNames={{
                    label: "text-[#E8BCB8]",
                    input: "text-[#E8BCB8]",
                    inputWrapper: "bg-[#44174E] border-[#862249]",
                  }}
                />
                <Select
                  label="Time Zone"
                  defaultSelectedKeys={["asia-manila"]}
                  classNames={{
                    label: "text-[#E8BCB8]",
                    trigger: "bg-[#44174E] border-[#862249] text-[#E8BCB8]",
                  }}
                >
                  <SelectItem key="asia-manila">Asia/Manila (UTC+8)</SelectItem>
                  <SelectItem key="utc">UTC</SelectItem>
                </Select>
                <Select
                  label="Language"
                  defaultSelectedKeys={["en"]}
                  classNames={{
                    label: "text-[#E8BCB8]",
                    trigger: "bg-[#44174E] border-[#862249] text-[#E8BCB8]",
                  }}
                >
                  <SelectItem key="en">English</SelectItem>
                  <SelectItem key="fil">Filipino</SelectItem>
                </Select>
              </div>

              <div className="space-y-4">
                <Slider
                  label="Recording Quality"
                  step={1}
                  minValue={1}
                  maxValue={10}
                  defaultValue={8}
                  className="max-w-md"
                  classNames={{
                    label: "text-[#E8BCB8]",
                    value: "text-[#ED9E59]",
                    track: "bg-[#44174E]",
                    filler: "bg-[#ED9E59]",
                  }}
                />
                <Slider
                  label="Data Retention (days)"
                  step={1}
                  minValue={7}
                  maxValue={90}
                  defaultValue={30}
                  className="max-w-md"
                  classNames={{
                    label: "text-[#E8BCB8]",
                    value: "text-[#ED9E59]",
                    track: "bg-[#44174E]",
                    filler: "bg-[#ED9E59]",
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
