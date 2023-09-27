from xml.etree import ElementTree as ET
import click
from click import Context, Path
import os
import copy


def read_xyn(locations_file: str) -> tuple[list, ...]:
    """Read a .xyn locations config file from D-FlowFM.

    Parameters
    ----------
    locations_file: str
        The path to the .xyn file.

    Returns
    -------
    list[float]
        The x coordinates (longitudes) of the stations.
    list[float]
        The y coordinates (latitudes) of the stations.
    list[str]
        The list of names for each station.
    """

    xs = []
    ys = []
    stations = []
    with open(locations_file, "r") as file:
        for line in file:
            x, y, name = line.split()
            xs.append(float(x))
            ys.append(float(y))
            stations.append(name.strip("'"))
    return xs, ys, stations


@click.group()
@click.option(
    "--locs-file",
    type=Path(exists=True),
    help="The .xyn file of selected outputs.",
)
@click.option(
    "--noos-folder",
    type=str,
    help="The folder that contains the .noos files inside stochObserver.",
)
@click.option(
    "-o",
    "--output-file",
    type=str,
    help="The noos observer XML file to write the configuration.",
)
@click.option(
    "--template",
    type=Path(exists=True),
    help="The location of the associated template XML file.",
)
@click.pass_context
def main(
    ctx: Context, locs_file: str, noos_folder: str, output_file: str, template: str
) -> None:
    """Read info needed for all XML operations."""

    ctx.ensure_object(dict)
    xs, ys, stations = read_xyn(locs_file)

    ctx.obj["xs"] = xs
    ctx.obj["ys"] = ys
    ctx.obj["stations"] = stations
    ctx.obj["noos_folder"] = noos_folder
    ctx.obj["output_file"] = output_file
    ctx.obj["template"] = template


@main.command()
@click.option(
    "-s",
    "--observation-std",
    type=float,
    default=0.05,
    help="The standard deviation of the observations.",
)
@click.argument(
    "noosfiles",
    nargs=-1,
    type=Path(exists=True),
)
@click.pass_context
def make_noos_observer(
    ctx: Context, observation_std: float, noosfiles: list[str]
) -> None:
    """Creates the stochObserver configuration file for the given noos files."""

    ET.register_namespace("", "http://www.openda.org")

    stations = ctx.obj["stations"]

    # Join path noos_folder/noosfile.noos, as inside stochObserver
    noosfiles = [
        os.path.join(ctx.obj["noos_folder"], os.path.basename(noosfile))
        for noosfile in noosfiles
    ]

    tree = ET.parse(ctx.obj["template"])
    root = tree.getroot()
    child = root[0]
    new_child = child

    for i, noosfile in enumerate(noosfiles):
        new_child.attrib["location"] = stations[i]
        new_child.attrib["standardDeviation"] = str(observation_std)
        new_child.text = noosfile
        root.append(new_child)
        new_child = copy.deepcopy(new_child)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=1)
    tree.write(ctx.obj["output_file"], encoding="utf-8", xml_declaration=True)


@main.command()
@click.option(
    "--npart",
    type=int,
    default=20,
    help="The number of model partitions.",
)
@click.argument(
    "noosfiles",
    nargs=-1,
    type=Path(exists=True),
)
@click.pass_context
def make_model(ctx: Context, npart: int, noosfiles: list[str]) -> None:
    """Creates the model XML configuration file for the given noos files and number of
    partitions."""

    ET.register_namespace("", "http://www.openda.org")

    stations = ctx.obj["stations"]
    npart = ctx.obj["npart"]

    # Join path noos_folder/noosfile.noos, as inside stochObserver
    noosfiles = [
        os.path.join(ctx.obj["noos_folder"], os.path.basename(noosfile))
        for noosfile in noosfiles
    ]

    tree = ET.parse(ctx.obj["template"])
    root = tree.getroot()

    # Fix alias (npart)
    for child in root[1]:
        if child.attrib["key"] == "npart":
            child.attrib["value"] = str(npart)
            break

    # Fix exchange items (restart)
    new_child = copy.deepcopy(root[3][0])
    for i in range(npart):
        rst_id = str(i).zfill(4)
        var = f"s1_{rst_id}"
        new_child.attrib["id"] = var
        new_child.attrib["ioObjectId"] = "rstfile"
        new_child.attrib["elementId"] = var
        root[3].append(new_child)
        new_child = copy.deepcopy(new_child)

    # Waterlevels
    new_child = copy.deepcopy(root[3][0])
    for station in stations:
        var = f"{station}.waterlevel"
        new_child.attrib["id"] = var
        new_child.attrib["ioObjectId"] = "averaged_hisfile"
        new_child.attrib["elementId"] = var
        root[3].append(new_child)
        new_child = copy.deepcopy(new_child)

    # Restart info
    new_child = copy.deepcopy(root[5][0])
    root[5].remove(root[5][0])
    for i in range(npart):
        rst_id = str(i).zfill(4)
        var = f"gtsm_fine_{rst_id}_00000000_000000_rst.nc"
        new_child.text = var
        root[5].append(new_child)
        new_child = copy.deepcopy(new_child)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=1)
    tree.write(ctx.obj["output_file"], encoding="utf-8", xml_declaration=True)


@main.command()
@click.option(
    "--npart",
    type=int,
    default=20,
    help="The number of model partitions.",
)
@click.argument(
    "noosfiles",
    nargs=-1,
    type=Path(exists=True),
)
@click.pass_context
def make_stoch_model(ctx: Context, npart: int, noosfiles: list[str]) -> None:
    """Creates the stochastic model XML configuration file for the given noos files and
    number of partitions."""

    ET.register_namespace("", "http://www.openda.org")

    stations = ctx.obj["stations"]

    # Join path noos_folder/noosfile.noos, as inside stochObserver
    noosfiles = [
        os.path.join(ctx.obj["noos_folder"], os.path.basename(noosfile))
        for noosfile in noosfiles
    ]

    tree = ET.parse(ctx.obj["template"])
    root = tree.getroot()

    # Fix state items
    local_root = root[1][0]
    new_child = copy.deepcopy(local_root[1])
    local_root.remove(local_root[1])
    for i in range(npart):
        rst_id = str(i).zfill(4)
        var = f"s1_{rst_id}"
        new_child.attrib["id"] = var
        local_root.append(new_child)
        new_child = copy.deepcopy(new_child)

    # Fix predictor items
    local_root = root[1][1]
    new_child = copy.deepcopy(local_root[0])
    local_root.remove(local_root[0])
    for station in stations:
        var = f"{station}.waterlevel"
        new_child.attrib["id"] = var
        local_root.append(new_child)
        new_child = copy.deepcopy(new_child)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=1)
    tree.write(ctx.obj["output_file"], encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
