public class Game_Stats
{
    private static int hp_0 = 0, hp_1 = 0;
    private static string ip = "None";

    public static string IP {
        get
        {
            return ip;
        }
        set
        {
            ip = value;
        }
    }

    public static int HP_0{
        get
        {
            return hp_0;
        }
        set
        {
            hp_0 = value;
        }
    }

    public static int HP_1
    {
        get
        {
            return hp_1;
        }
        set
        {
            hp_1 = value;
        }
    }
}
