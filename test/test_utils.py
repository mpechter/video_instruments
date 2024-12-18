#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Dec 18, 2024

@author: mpechter

"""
from single_keyboard import wrist_is_visible, wrist_is_on_keyboard, get_midi_velocity


class TestThereminUtils:

    def test_if_wrist_visible(self):
        assert wrist_is_visible(0.60) == True

    def test_wrist_not_visible(self):
        assert wrist_is_visible(0.57) == False

    def test_wrist_on_keyboard(self):
        assert wrist_is_on_keyboard(0.40) == True

    def test_wrist_not_on_keyboard(self):
        assert wrist_is_on_keyboard(0.60) == False
