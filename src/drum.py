#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import rtmidi

VELOCITY_BUFFER = 0.01
NOTE_HYSTERESIS = 1
STRIKE_THRESHOLD = 5
READY_TO_STRIKE_THRESHOLD = 1
STRIKE_LEVEL = 0.85


class WristTracker:
    def __init__(self, side: str):
        self.side = side
        self.current_note = None
        self.set = False

    def reset_counters(self):
        self.set = False


class Drum:
    def __init__(self):
        self.initialize_midi()
        self.initialize_mediapose()
        self.initialize_webcam()
        self.set_detection_thresholds()
        self.left_wrist = WristTracker("left")
        self.right_wrist = WristTracker("right")

    def initialize_midi(self):
        self.midi_out = rtmidi.MidiOut()
        available_ports = self.midi_out.get_ports()
        if available_ports:
            self.midi_out.open_port(0)
        else:
            self.midi_out.open_virtual_port("Drum")

    def set_detection_thresholds(self):
        self.top_threshold = 0.25
        self.bottom_threshold = 0.99
        # self.right_threshold = 0.25
        # self.left_threshold = 0.75

    def initialize_mediapose(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def initialize_webcam(self):
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def change_is_less_than_hysteresis(
        self, normalized_note: int, current_note
    ) -> bool:
        note_diff = abs(normalized_note - current_note)
        return note_diff < NOTE_HYSTERESIS

    def get_midi_note(self, x_pos: int, current_note: int) -> int:
        # x_pos from MediaPipe is already 0-1 normalized across view width
        if current_note is not None:
            # Keep hysteresis check for stability
            if abs(x_pos - current_note) < NOTE_HYSTERESIS:
                return current_note

        mirrored_x = 100 * (1 - x_pos)

        if mirrored_x > 80:
            return 51
        elif mirrored_x > 50:
            return 42
        elif mirrored_x > 20:
            return 38
        else:
            return 36

    def play_note(self, note, wrist_tracker):
        self.stop_note(wrist_tracker)
        self.midi_out.send_message([0x90, note, 100])
        self.stop_note(wrist_tracker)  # Since it is a drum, we quickly stop the note.

    def stop_note(self, wrist_tracker):
        if wrist_tracker.current_note is not None:
            self.midi_out.send_message([0x80, wrist_tracker.current_note, 0])

    def process_wrist(self, wrist, wrist_tracker):
        if wrist and wrist_is_visible(wrist.visibility):
            if wrist_tracker.set and wrist.y >= STRIKE_LEVEL:
                note = self.get_midi_note(wrist.x, wrist_tracker.current_note)
                self.play_note(note, wrist_tracker)
                wrist_tracker.current_note = note
                wrist_tracker.reset_counters()
            elif wrist.y < STRIKE_LEVEL:
                wrist_tracker.set = True

    def run(self):
        try:
            while self.webcam.isOpened():
                success, image = self.webcam.read()
                if not success:
                    print("Failed to get webcam frame")
                    break

                left_wrist = get_wrist_object(image, self, "left")
                right_wrist = get_wrist_object(image, self, "right")

                self.process_wrist(left_wrist, self.left_wrist)
                self.process_wrist(right_wrist, self.right_wrist)

        finally:
            self.cleanup()

    def cleanup(self):
        self.stop_note(self.left_wrist)
        self.stop_note(self.right_wrist)
        self.webcam.release()
        cv2.destroyAllWindows()
        self.midi_out.close_port()


def wrist_is_on_trigger_level(y_position: int) -> bool:
    return y_position > 0.4


def wrist_is_visible(visibility: int) -> bool:
    return visibility > 0.58


def get_wrist_object(image, drum, side):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = drum.pose.process(image_rgb)

    if results.pose_landmarks:
        landmark = (
            drum.mp_pose.PoseLandmark.LEFT_WRIST
            if side == "left"
            else drum.mp_pose.PoseLandmark.RIGHT_WRIST
        )
        return results.pose_landmarks.landmark[landmark]
    return False


if __name__ == "__main__":
    controller = Drum()
    controller.run()
