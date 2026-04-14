import { Hotbar } from "./components/Hotbar";
import { ModeIndicator } from "./components/ModeIndicator";
import { Inventory } from "./components/Inventory";
import { ColorPicker } from "./components/ColorPicker";
import { ToastContainer } from "./components/Toast";
import { PauseMenu } from "./components/PauseMenu";
import { DebugOverlay } from "./components/DebugOverlay";

export default function App() {
  return (
    <>
      <Hotbar />
      <ModeIndicator />
      <Inventory />
      <ColorPicker />
      <PauseMenu />
      <DebugOverlay />
      <ToastContainer />
    </>
  );
}
