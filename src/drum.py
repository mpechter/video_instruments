#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jan 10, 2024

@author: mpechter

"""

import cv2
import mediapipe as mp
import rtmidi

VELOCITY_BUFFER = 0.01  # Any velocity below this is ignored to reduce shakiness.
NOTE_HYSTERESIS = 1  # To prevent note flickering.
STRIKE_THRESHOLD = 2  # How many cycles of positive velocity before a note is played.
READY_TO_STRIKE_THRESHOLD = 2


class Drum:
    def __init__(self):
        self.initialize_midi()
        self.initialize_mediapose()
        self.initialize_webcam()
        self.set_detection_thresholds()
        self.velocity = 0
        self.left_release_counter = 0
        self.left_current_note = None
        self.left_strike_counter = 0

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
        self.right_threshold = 0.25
        self.left_threshold = 0.75

    def initialize_mediapose(self):
        self.mp_pose = mp.solutions.pose
        # self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def initialize_webcam(self):
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def normalize_note(self, x_pos: int) -> int:

        normalized_within_thresholds = (x_pos - self.right_threshold) / (
            self.left_threshold - self.right_threshold
        )
        clamped_to_0_1 = max(0, min(1, normalized_within_thresholds))
        inverted = 1 - clamped_to_0_1

        return inverted

    def change_is_less_than_hysteresis(
        self, normalized_note: int, current_note
    ) -> bool:

        note_diff = abs(normalized_note - current_note)
        return note_diff < NOTE_HYSTERESIS

    def get_midi_note(self, x_pos: int, current_note: int) -> int:
        """Convert vertical position to MIDI note."""
        normalized_note = self.normalize_note(x_pos)

        if current_note is not None:
            if self.change_is_less_than_hysteresis(normalized_note, current_note):
                return current_note

        if normalized_note < 0.25:
            return 36
        elif normalized_note < 0.5:
            return 38
        elif normalized_note < 0.75:
            return 42
        else:
            return 48

    def play_note(self, note):

        self.stop_note()
        self.midi_out.send_message([0x90, note, 100])
        self.stop_note()  # Since it is a drum, we quickly stop the note.

    def stop_note(self):

        if self.current_note is not None:
            self.midi_out.send_message([0x80, self.current_note, 0])

    def get_velocity(self, left_wrist):

        if self.previous_y is not None:
            self.velocity = left_wrist.y - self.previous_y
        else:
            self.velocity = 0

        if abs(self.velocity) < VELOCITY_BUFFER:
            self.velocity = 0

    def run(self):
        """Main loop for the application."""
        try:
            self.previous_y = None
            self.previous_velocity = None
            self.was_visible_last_cycle = False
            while self.webcam.isOpened():
                success, image = self.webcam.read()
                if not success:
                    print("Failed to get webcam frame")
                    break

                left_wrist = get_left_wrist_object(image, self)
                # cv2.imshow("Pose MIDI Controller", image)
                if left_wrist and wrist_is_visible(left_wrist.visibility):
                    self.get_velocity(left_wrist)

                    if self.velocity > 0:
                        self.left_strike_counter += 1
                    else:
                        self.left_release_counter += 1

                    if (
                        self.left_strike_counter > STRIKE_THRESHOLD
                        and wrist_is_on_trigger_level(left_wrist.y)
                        and self.left_release_counter > READY_TO_STRIKE_THRESHOLD
                    ):
                        note = self.get_midi_note(left_wrist.x, self.left_current_note)
                        self.play_note(note)
                        self.left_current_note = note
                        self.left_strike_counter = 0
                        self.left_release_counter = 0

                self.previous_y = left_wrist.y if left_wrist else self.previous_y
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if self.current_note is not None:
            self.midi_out.send_message([0x80, self.current_note, 0])
        self.webcam.release()
        cv2.destroyAllWindows()
        self.midi_out.close_port()


def wrist_is_on_trigger_level(y_position: int) -> bool:

    return 0.90 < y_position


def wrist_is_visible(visibility: int) -> bool:

    return visibility > 0.58


def get_left_wrist_object(image, theremin):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = theremin.pose.process(image_rgb)

    if results.pose_landmarks:
        return results.pose_landmarks.landmark[theremin.mp_pose.PoseLandmark.LEFT_WRIST]
    else:
        return False


if __name__ == "__main__":
    controller = Drum()
    controller.run()
