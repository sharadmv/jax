# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module for building interfaces in Colab."""
from __future__ import annotations

import abc
import dataclasses
import functools
import sys
import uuid

from typing import Any, Dict, List, Union

IS_COLAB_ENABLED = True

from IPython import display

class DOMElement(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def render(self):
    pass


Element = Union[DOMElement, str]


class DynamicDOMElement(DOMElement):
  """A DOM element that can be mutated."""

  @abc.abstractmethod
  def render(self):
    pass

  @abc.abstractmethod
  def append(self, child: DOMElement):
    pass

  @abc.abstractmethod
  def update(self, elem: DOMElement):
    pass

  @abc.abstractmethod
  def clear(self):
    pass

@dataclasses.dataclass
class DynamicDiv(DynamicDOMElement):
  """A `div` that can be edited."""
  _uuid: str = dataclasses.field(init=False)
  _root_elem: DOMElement = dataclasses.field(init=False)
  elem: Union[DOMElement, str]

  def __post_init__(self):
    self._uuid = str(uuid.uuid4())
    self._rendered = False
    self._root_elem = div(self.elem)

  def _as_html(self):
    return display.HTML(self._root_elem.html())

  def render(self):
    if self._rendered:
      raise ValueError("Can't call `render` twice.")
    self._handle = display.display(self._as_html(), display_id=self._uuid)
    self._rendered = True

  def append(self, child: DOMElement):
    if not self._rendered:
      self.render()
    children = self._root_elem.children
    self._root_elem = div(*children, child)
    self._handle.update(self._as_html())

  def update(self, elem: DOMElement):
    if not self._rendered:
      self.render()
    self._root_elem = div(elem)
    self._handle.update(self._as_html())

  def clear(self):
    if not self._rendered:
      self.render()
    self._root_elem = div()
    self._handle.update(self._as_html())


@dataclasses.dataclass
class StaticDOMElement(DOMElement):
  """An immutable DOM element."""
  _uuid: str = dataclasses.field(init=False)
  name: str
  children: List[Union[str, DOMElement]]
  attrs: Dict[str, str]

  def __repr_html__(self):
    return self.html()

  def html(self):
    attr_str = ""
    if self.attrs:
      attr_str = " " + (" ".join(
          [f"{key}=\"{value}\"" for key, value in self.attrs.items()]))
    children = []
    children = "\n".join([str(c) for c in self.children])
    return f"<{self.name}{attr_str}>{children}</{self.name}>"

  def render(self):
    return display.display(display.HTML(self.html()))

  def attr(self, key: str) -> str:
    return self.attrs[key]

  def __str__(self):
    return self.html()

  def __repr__(self):
    return self.html()

  def append(self, child: DOMElement) -> DOMElement:
    return dataclasses.replace(self, children=[*self.children, child])

  def replace(self, **kwargs) -> DOMElement:
    return dataclasses.replace(self, **kwargs)


def _style_dict_to_str(style_dict: Dict[str, Any]) -> str:
  return " ".join([f"{k}: {v};" for k, v in style_dict.items()])


def dynamic(elem: StaticDOMElement) -> DynamicDiv:
  return DynamicDiv(elem)


def _make_elem(tag: str, *children: Element, **attrs) -> StaticDOMElement:
  """Helper function for making DOM elements."""
  return StaticDOMElement(tag, list(children), attrs)


code = functools.partial(_make_elem, "code")
div = functools.partial(_make_elem, "div")
li = functools.partial(_make_elem, "li")
ol = functools.partial(_make_elem, "ol")
pre = functools.partial(_make_elem, "pre")
progress = functools.partial(_make_elem, "progress")
span = functools.partial(_make_elem, "span")


def css(text: str) -> StaticDOMElement:
  return StaticDOMElement("style", [text], {})


def style(*args, **kwargs):
  return _style_dict_to_str(dict(*args, **kwargs))
