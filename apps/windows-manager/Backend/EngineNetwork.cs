using System.Net;
using System.Net.NetworkInformation;
using System.Net.Sockets;

namespace NInfer.Manager;

internal sealed record LanInterface(string Id, NetworkInterfaceType Type, bool Up,
    bool Virtual, bool HasGateway, IReadOnlyList<IPAddress> Addresses);

internal static class EngineNetwork
{
    public static string ApiBase(string host, int port)
    {
        var address = host == "0.0.0.0" ? SelectLanAddress(ReadInterfaces())?.ToString()
            ?? throw new InvalidOperationException("未找到可用的局域网 IPv4 地址。请连接以太网或 Wi-Fi，或选择 127.0.0.1。 / No usable LAN IPv4 address. Connect Ethernet or Wi-Fi, or select 127.0.0.1.") : host;
        return $"http://{address}:{port}/v1";
    }

    internal static IPAddress? SelectLanAddress(IEnumerable<LanInterface> interfaces) => interfaces
        .Where(nic => nic.Up && !nic.Virtual && nic.Type is NetworkInterfaceType.Ethernet or NetworkInterfaceType.Wireless80211)
        .OrderByDescending(nic => nic.HasGateway)
        .ThenBy(nic => nic.Type == NetworkInterfaceType.Ethernet ? 0 : 1)
        .ThenBy(nic => nic.Id, StringComparer.Ordinal)
        .SelectMany(nic => nic.Addresses.Where(IsUsableIPv4).OrderBy(address =>
            Convert.ToHexString(address.GetAddressBytes()), StringComparer.Ordinal))
        .FirstOrDefault();

    private static bool IsUsableIPv4(IPAddress address)
    {
        if (address.AddressFamily != AddressFamily.InterNetwork) return false;
        var bytes = address.GetAddressBytes();
        return bytes[0] is > 0 and < 224 && bytes[0] != 127 && !(bytes[0] == 169 && bytes[1] == 254);
    }

    private static IEnumerable<LanInterface> ReadInterfaces()
    {
        foreach (var nic in NetworkInterface.GetAllNetworkInterfaces())
        {
            var properties = nic.GetIPProperties();
            // Windows presents several software adapters as Ethernet; they are not a useful LAN URL.
            var description = nic.Name + " " + nic.Description;
            var virtualAdapter = new[] { "virtual", "hyper-v", "vmware", "wsl", "docker", "loopback", "tunnel", "tap-windows", "tailscale", "zerotier", "wireguard" }
                .Any(word => description.Contains(word, StringComparison.OrdinalIgnoreCase));
            yield return new(nic.Id, nic.NetworkInterfaceType, nic.OperationalStatus == OperationalStatus.Up,
                virtualAdapter, properties.GatewayAddresses.Any(gateway => IsUsableIPv4(gateway.Address)),
                properties.UnicastAddresses.Where(address => address.DuplicateAddressDetectionState == DuplicateAddressDetectionState.Preferred)
                    .Select(address => address.Address).ToArray());
        }
    }
}
