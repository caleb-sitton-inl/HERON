from typing import Any
import re
import xml.etree.ElementTree as ET


def parse_xpath(xpath: str):
  """
  Parses an XPath with possible attributes and text content into a list of dictionaries.
  Each dictionary contains "tag" and "attrib" keys.
  @ In, xpath, str, an XPath describing a tree of nodes
  @ Out, nodes, list[dict[str, str]], list of dicts describing the nodes of the tree described by the XPath
  """
  # Regular expression to match node with optional attributes
  # Regex for XPaths created using AiVA
  node_pattern = re.compile(r"(?P<tag>[^/\[]+)(\[@(?P<attrib>[^=]+)=(?P<quote>['\"])(?P<value>[^'\"]+)(?P=quote)\])?")

  nodes = []
  for match in node_pattern.finditer(xpath.strip('/')):
    tag = match.group('tag')
    attrib = {match.group('attrib'): match.group('value')} if match.group('attrib') else {}
    nodes.append({'tag': tag, 'attrib': attrib})

  return nodes

def add_node_to_tree(child_node: ET.Element, parent_path: str, root: ET.Element) -> None:
  """
  Adds a child XML node to a parent node specified by an XPath.
  Creates any necessary intermediate nodes with attributes and text if they do not exist.

  @ In, child_node, ET.Element, object representing the child node to be added
  @ In, xpath, str, string representing the XPath to the parent node
  @ In, root, ET.Element,
  """
  # Parse the XPath into nodes with tag, attributes, and text
  path_parts = parse_xpath(parent_path)

  # Start from the root
  current_node = root

  for node_xpath, parsed in zip(parent_path.strip("/").split("/"), path_parts):
    # Find the next node by xpath
    next_node = current_node.find(node_xpath)
    if next_node is None:
      # Make the node with the parsed tag and attributes
      next_node = ET.SubElement(current_node, parsed["tag"], attrib=parsed["attrib"])
    current_node = next_node

  # Append the child node to the current (parent) node
  current_node.append(child_node)

def stringify_node_values(node):
    """
    Ensure that XML node attribute and text values are strings before trying to express the XML tree as a string
    that is written to file. Traverses the XML tree recursively.
    @ In, node, ET.Element, node to begin traversal at
    @ Out, None
    """
    for k, v in node.attrib.items():
      node.attrib[k] = _to_string(v)

    text = node.text
    if text is not None and not isinstance(text, str):
      node.text = _to_string(node.text)

    # Traverse children
    for child in node:
        stringify_node_values(child)

def _to_string(val: Any, delimiter: str = ", ") -> str:
  """
  Express the provided object as an appropriate string for a node text. If it's a string or numeric type,
  those can easily be expressed as strings. If the object is list-like (list, tuple, set, numpy array, etc.),
  the object elements are joined in a comma-separated list. Mappings like dictionaries are not supported.

  @ In, val, Any, value to be converted to string for node text
  @ In, delimiter, str, string used to delimit iterator items
  @ out, text, str, string representative of the value
  """
  if isinstance(val, str):
    return val
  elif isinstance(val, dict):
    raise TypeError(f"Unable to convert dicts to node text. Received type '{type(val)}'.")
  elif hasattr(val, "__iter__") and not isinstance(val, str):  # lists, ListWrapper, sets, numpy arrays, etc.
    # return as comma-separated string
    return delimiter.join([str(v) for v in val])
  else:  # numeric types and other objects, including RavenSnippet objects
    return str(val)

def find_node(parent: ET.Element, tag: str, make_if_missing: bool = True) -> ET.Element | None:
  """
  Find the first node with tag in parent
  @ In, parent, ET.Element, parent node
  @ In, tag, str, tag of node to find
  @ In, make_if_missing, bool, optional, make a node with the given tag if one is not found
  @ Out, node, ET.Element, the found or created node
  """
  node = parent.find(tag)
  if node is not None or not make_if_missing:
    return node

  # We need to make the child node before returning. There may be an intermediate subtree
  # described in the tag, so we need to make any intermediate nodes along the way to the
  # final child node.
  node = parent
  raw_splits = [p.strip() for p in tag.split("/")]
  for child_xpath, child_params in zip(raw_splits, parse_xpath(tag)):
    next_node = node.find(child_xpath)
    if next_node is None:
      next_node = ET.SubElement(node, child_params["tag"], child_params["attrib"])
    node = next_node

  return node

def merge_trees(left: ET.Element, right: ET.Element, *, overwrite: bool = True, match_attrib: bool = True) -> ET.Element:
  """
  Merge "right" tree into "left" tree. Equivalent nodes are defined by having equal tags and attributes. If overwrite
  is True, the attributes and text of an element of "left" will be overwritten by the values in a matching node in
  "right", if present. Equality is determined by either just the tag (match_attrib=False) or the tag and all attribute
  values (match_attrib=False). If overwrite is False, all leaf nodes from right are added to left in the matching
  location.

  @ In, left, ET.Element, the first tree
  @ In, right, ET.Element, the second tree
  @ In overwrite, bool, optional, if duplicate nodes
  """
  def find_matching_node(node, candidates):
    for candidate in candidates:
      if node.tag != candidate.tag:
        continue
      if match_attrib:
        if node.attrib == candidate.attrib:
          return candidate
      else:
        return candidate
    return None

  def merge_nodes(left_node, right_node):
    matching_node = find_matching_node(right_node, left_node)
    if matching_node is not None:
      if overwrite:
        matching_node.attrib.update(right_node.attrib)
        matching_node.text = right_node.text
      for r_child in right_node:
        merge_nodes(matching_node, r_child)
    else:
      new_elem = ET.SubElement(left_node, right_node.tag, right_node.attrib)
      new_elem.text = right_node.text
      for r_child in right_node:
        merge_nodes(new_elem, r_child)

  for r_child in right:
    merge_nodes(left, r_child)

  return left
