#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Dec 1, 2024

@author: mpechter

"""

import cv2
import mediapipe as mp
import rtmidi


class VideoTheremin:
    def __init__(self):
        self.initialize_midi()
        self.initialize_mediapose()
        self.initialize_webcam()
        self.set_midi_defaults()
        self.set_musical_defaults()
        self.set_detection_thresholds()
        self.velocity = 0
        self.release_threshold = 0.05  # Soft threshold for note release
        self.note_state = "idle"  # Track note state explicitly
        self.release_counter = 0
        self.max_release_count = 10  #

    def initialize_midi(self):
        self.midi_out = rtmidi.MidiOut()
        available_ports = self.midi_out.get_ports()
        if available_ports:
            self.midi_out.open_port(0)
        else:
            self.midi_out.open_virtual_port("Video Theremin")

        # Initialize MediaPipe Pose

    def set_musical_defaults(self):
        self.scales = {
            "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "pentatonic": [0, 2, 4, 7, 9],
            "blues": [0, 3, 5, 6, 7, 10],
        }
        self.current_scale = "major"
        self.root_note = 0  # C

    def set_detection_thresholds(self):
        self.top_threshold = 0.25
        self.bottom_threshold = 0.99
        self.right_threshold = 0.25
        self.left_threshold = 0.75

    def set_midi_defaults(self):
        self.current_note = None
        self.min_note = 48  # C3
        self.max_note = 60  # C4

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

    def quantize_to_scale(self, note):
        """Quantize a MIDI note number to the current scale."""
        # Get the octave and note within octave
        octave = (note - self.min_note) // 12
        note_in_octave = (note - self.min_note) % 12

        # Find the closest note in the scale
        scale_notes = self.scales[self.current_scale]
        scale_note = min(scale_notes, key=lambda x: abs(x - note_in_octave))

        # Reconstruct the MIDI note number
        quantized_note = self.min_note + (octave * 12) + scale_note

        # Ensure we stay within our note range
        return max(self.min_note, min(quantized_note, self.max_note))

    def get_midi_note(self, y_pos):
        """Convert vertical position to MIDI note."""
        # Normalize y_pos to 0-1 range within thresholds
        y_normalized = (y_pos - self.top_threshold) / (
            self.bottom_threshold - self.top_threshold
        )
        y_normalized = max(0, min(1, y_normalized))  # Clamp to 0-1

        # Invert y_pos (higher position = higher pitch)
        y_normalized = 1 - y_normalized

        # Map to MIDI note range
        raw_note = int(y_normalized * (self.max_note - self.min_note) + self.min_note)

        return self.quantize_to_scale(raw_note)

    def get_note_name(self, note):
        """Convert MIDI note number to note name."""
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (note // 12) - 1
        note_name = note_names[note % 12]
        return f"{note_name}{octave}"

    def play_note(self, note):

        if self.current_note != note:
            self.stop_note()
            self.midi_out.send_message([0x90, note, self.midi_velocity])
            self.current_note = note
        if self.note_state == "idle":
            self.stop_note()
            self.midi_out.send_message([0x90, note, self.midi_velocity])
            self.current_note = note

    def stop_note(self):

        if self.current_note is not None:
            self.midi_out.send_message([0x80, self.current_note, 0])

    def get_velocity(self, left_wrist):

        if self.previous_x is not None:
            self.velocity = self.previous_x - left_wrist.x
        else:
            self.velocity = 0

        if abs(self.velocity) < 0.01:
            self.velocity = 0

        self.midi_velocity = get_midi_velocity(self.velocity)

    def run(self):
        """Main loop for the application."""
        try:
            self.previous_x = None
            self.previous_velocity = None
            self.was_visible_last_cycle = False
            while self.webcam.isOpened():
                success, image = self.webcam.read()
                if not success:
                    print("Failed to get webcam frame")
                    break

                left_wrist = get_left_wrist_object(image, self)
                # cv2.imshow("Pose MIDI Controller", image)

                if (
                    left_wrist
                    and wrist_is_visible(left_wrist.visibility)
                    and wrist_is_on_keyboard(left_wrist.x)
                ):
                    self.get_velocity(left_wrist)
                    print(self.velocity)

                    if self.velocity > 0:
                        note = self.get_midi_note(left_wrist.y)
                        self.play_note(note)
                        self.note_state = "playing"
                        self.release_counter = 0
                    elif self.velocity < 0:
                        self.release_counter += 1

                    if self.note_state == "playing":
                        if self.release_counter > 3:
                            self.stop_note()
                            print("pt 1")
                            self.note_state = "idle"

                        if self.release_counter > self.max_release_count:
                            self.stop_note()
                            print("pt 2")
                            self.note_state = "idle"

                else:
                    if self.note_state == "playing":
                        self.release_counter += 1

                        if self.release_counter > self.max_release_count:
                            self.stop_note()
                            print("pt 3")
                            self.note_state = "idle"

                self.previous_x = left_wrist.x if left_wrist else self.previous_x
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if self.current_note is not None:
            self.midi_out.send_message([0x80, self.current_note, 0])
        self.webcam.release()
        cv2.destroyAllWindows()
        self.midi_out.close_port()


def wrist_is_on_keyboard(x_position: int) -> bool:

    return 0.50 < x_position < 0.75


def wrist_is_visible(visibility: int) -> bool:

    return visibility > 0.5


def get_left_wrist_object(image, theremin):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = theremin.pose.process(image_rgb)

    if results.pose_landmarks:
        return results.pose_landmarks.landmark[theremin.mp_pose.PoseLandmark.LEFT_WRIST]
    else:
        return False


def get_midi_velocity(
    raw_velocity: float,
):
    """
    Compresses velocity to boost lows and soften highs using a power curve.

    Args:
        input_velocity (float): The raw velocity input (e.g., wrist movement speed).

    Returns:
        int: MIDI velocity (0-127) after applying compression.
    """

    power = 0.8
    min_velocity = 60
    max_velocity = 100

    if raw_velocity < 0:
        return raw_velocity

    input_velocity = raw_velocity * 5000

    if input_velocity <= min_velocity:
        return min_velocity

    normalized_velocity = (input_velocity - min_velocity) / (
        max_velocity - min_velocity
    )
    velocity_as_percent = normalized_velocity**power

    velocity_as_portion_of_max = int(velocity_as_percent * max_velocity)

    clipped_to_range = max(min_velocity, min(max_velocity, velocity_as_portion_of_max))

    return clipped_to_range


if __name__ == "__main__":
    controller = VideoTheremin()
    controller.run()
