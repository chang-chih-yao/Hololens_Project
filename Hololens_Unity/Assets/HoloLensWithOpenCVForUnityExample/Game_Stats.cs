public class Game_Stats
{
    private static int hp_0 = 0, hp_1 = 0;
    private static float pre_z = -35f;
    private static float height = 175f;
    private static string ip = "192.168.";
    private static string playerID = "None";
    private static bool demo = false;
    private static bool debug = false;
    private static bool draw_skeleton = true;

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

    public static int HP_0
    {
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

    public static string PlayerID
    {
        get
        {
            return playerID;
        }
        set
        {
            playerID = value;
        }
    }

    public static bool DEMO
    {
        get
        {
            return demo;
        }
        set
        {
            demo = value;
        }
    }

    public static bool Debug
    {
        get
        {
            return debug;
        }
        set
        {
            debug = value;
        }
    }

    public static bool Draw_skeleton
    {
        get
        {
            return draw_skeleton;
        }
        set
        {
            draw_skeleton = value;
        }
    }

    public static float Pre_z
    {
        get
        {
            return pre_z;
        }
        set
        {
            pre_z = value;
        }
    }

    public static float Height
    {
        get
        {
            return height;
        }
        set
        {
            height = value;
        }
    }
}
