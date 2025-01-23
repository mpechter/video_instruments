#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import rtmidi

VELOCITY_BUFFER = 0.01
NOTE_HYSTERESIS = 1
STRIKE_THRESHOLD = 5
READY_TO_STRIKE_THRESHOLD = 1


class WristTracker:
    def __init__(self, side: str):
        self.side = side
        self.velocity = 0
        self.release_counter = 0
        self.current_note = None
        self.strike_counter = 0
        self.previous_y = None
        self.previous_velocity = None

    def get_velocity(self, wrist):
        if self.previous_y is not None:
            self.velocity = wrist.y - self.previous_y
        else:
            self.velocity = 0

        if abs(self.velocity) < VELOCITY_BUFFER:
            self.velocity = 0

        print(self.velocity)

    def update_position(self, wrist):
        self.previous_y = wrist.y if wrist else self.previous_y

    def should_play_note(self, wrist) -> bool:
        if self.velocity > 0:
            self.strike_counter += 1
        elif self.velocity <= 0:
            self.release_counter += 1

        return (
            self.strike_counter > STRIKE_THRESHOLD
            and wrist_is_on_trigger_level(wrist.y)
            and self.release_counter >= READY_TO_STRIKE_THRESHOLD
        )

    def reset_counters(self):
        self.strike_counter = 0
        self.release_counter = 0


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

    # def normalize_note(self, x_pos: int) -> int:
    #     normalized_within_thresholds = (x_pos - self.right_threshold) / (
    #         self.left_threshold - self.right_threshold
    #     )
    #     clamped_to_0_1 = max(0, min(1, normalized_within_thresholds))

    #     return 1 - clamped_to_0_1

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

        mirrored_x = 1 - x_pos

        if mirrored_x < 0.25:
            return 36
        elif mirrored_x < 0.5:
            return 38
        elif mirrored_x < 0.75:
            return 42
        else:
            return 48

    def play_note(self, note, wrist_tracker):
        self.stop_note(wrist_tracker)
        self.midi_out.send_message([0x90, note, 100])
        self.stop_note(wrist_tracker)  # Since it is a drum, we quickly stop the note.

    def stop_note(self, wrist_tracker):
        if wrist_tracker.current_note is not None:
            self.midi_out.send_message([0x80, wrist_tracker.current_note, 0])

    def process_wrist(self, wrist, wrist_tracker):
        if wrist and wrist_is_visible(wrist.visibility):
            wrist_tracker.get_velocity(wrist)

            if wrist_tracker.should_play_note(wrist):
                note = self.get_midi_note(wrist.x, wrist_tracker.current_note)
                self.play_note(note, wrist_tracker)
                print(wrist_tracker.side)
                wrist_tracker.current_note = note
                wrist_tracker.reset_counters()

            wrist_tracker.update_position(wrist)

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
    return 0.50 < y_position


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
